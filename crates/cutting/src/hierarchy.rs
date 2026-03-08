//! Precedence constraint DAG for cutting order.
//!
//! Builds a directed acyclic graph (DAG) that encodes which contours must
//! be cut before others:
//!
//! 1. **Interior before Exterior**: All holes within a part must be cut
//!    before the part's exterior boundary.
//! 2. **Inner parts before outer parts**: If part A is geometrically
//!    contained within part B's hole, part A must be cut first.
//!
//! # References
//!
//! - Dewil et al. (2016), Section 3.2: "Precedence constraints"

use crate::contour::{ContourId, ContourType, CutContour};

/// Precedence constraint DAG for cutting order.
#[derive(Debug, Clone)]
pub struct CuttingDag {
    /// Number of contours.
    num_contours: usize,
    /// Precedence edges: (before, after) — contour `before` must be cut before `after`.
    precedences: Vec<(ContourId, ContourId)>,
}

impl CuttingDag {
    /// Builds a precedence DAG from extracted contours.
    ///
    /// # Rules
    ///
    /// 1. For each part, all Interior contours precede the Exterior contour.
    /// 2. If a part's exterior is geometrically contained within another part's
    ///    hole, the inner part must be fully cut before the outer part's hole.
    ///    (This is detected by checking if the inner part's centroid lies within
    ///    the outer part's exterior contour.)
    pub fn build(contours: &[CutContour]) -> Self {
        let num_contours = contours.len();
        let mut precedences = Vec::new();

        // Rule 1: Interior holes before their parent Exterior
        // Group contours by (geometry_id, instance)
        let mut part_groups: std::collections::HashMap<(&str, usize), Vec<&CutContour>> =
            std::collections::HashMap::new();

        for contour in contours {
            part_groups
                .entry((contour.geometry_id.as_str(), contour.instance))
                .or_default()
                .push(contour);
        }

        for group in part_groups.values() {
            let exterior = group
                .iter()
                .find(|c| c.contour_type == ContourType::Exterior);
            let interiors: Vec<&&CutContour> = group
                .iter()
                .filter(|c| c.contour_type == ContourType::Interior)
                .collect();

            if let Some(ext) = exterior {
                for interior in &interiors {
                    precedences.push((interior.id, ext.id));
                }
            }
        }

        // Rule 2: Nested parts (part inside another part's boundary)
        // Check if any part's exterior centroid is inside another part's exterior
        let exteriors: Vec<&CutContour> = contours
            .iter()
            .filter(|c| c.contour_type == ContourType::Exterior)
            .collect();

        for i in 0..exteriors.len() {
            for j in 0..exteriors.len() {
                if i == j {
                    continue;
                }
                // Check if exterior[i]'s centroid is inside exterior[j]
                if u_nesting_core::geom::polygon::contains_point(
                    &exteriors[j].vertices,
                    exteriors[i].centroid,
                ) {
                    // Part i is inside part j -> all of part i must be cut before part j's exterior
                    let part_i_key = (exteriors[i].geometry_id.as_str(), exteriors[i].instance);
                    if let Some(part_i_contours) = part_groups.get(&part_i_key) {
                        for contour in part_i_contours {
                            precedences.push((contour.id, exteriors[j].id));
                        }
                    }
                }
            }
        }

        Self {
            num_contours,
            precedences,
        }
    }

    /// Returns the number of contours.
    pub fn num_contours(&self) -> usize {
        self.num_contours
    }

    /// Returns all precedence constraints as (before, after) pairs.
    pub fn precedences(&self) -> &[(ContourId, ContourId)] {
        &self.precedences
    }

    /// Checks if cutting contour `a` before contour `b` violates any precedence.
    ///
    /// Returns `true` if `b` must be cut before `a` (i.e., `a` before `b` is invalid).
    pub fn violates(&self, a: ContourId, b: ContourId) -> bool {
        // Check if there's a precedence (b, a) — meaning b must come before a
        // If so, putting a before b violates it
        self.precedences
            .iter()
            .any(|&(before, after)| before == b && after == a)
    }

    /// Returns the set of contours that must be cut before the given contour.
    pub fn predecessors(&self, contour_id: ContourId) -> Vec<ContourId> {
        self.precedences
            .iter()
            .filter(|(_, after)| *after == contour_id)
            .map(|(before, _)| *before)
            .collect()
    }

    /// Returns the set of contours that must be cut after the given contour.
    pub fn successors(&self, contour_id: ContourId) -> Vec<ContourId> {
        self.precedences
            .iter()
            .filter(|(before, _)| *before == contour_id)
            .map(|(_, after)| *after)
            .collect()
    }

    /// Checks if a given sequence respects all precedence constraints.
    pub fn is_valid_sequence(&self, sequence: &[ContourId]) -> bool {
        // Build position map: contour_id -> position in sequence
        let mut position: std::collections::HashMap<ContourId, usize> =
            std::collections::HashMap::new();
        for (pos, &id) in sequence.iter().enumerate() {
            position.insert(id, pos);
        }

        // Check all precedences
        for &(before, after) in &self.precedences {
            if let (Some(&pos_before), Some(&pos_after)) =
                (position.get(&before), position.get(&after))
            {
                if pos_before >= pos_after {
                    return false;
                }
            }
        }

        true
    }

    /// Produces a valid topological ordering of contour IDs.
    ///
    /// Uses Kahn's algorithm. Returns None if the DAG has a cycle.
    pub fn topological_sort(&self) -> Option<Vec<ContourId>> {
        let n = self.num_contours;
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<ContourId>> = vec![Vec::new(); n];

        for &(before, after) in &self.precedences {
            if before < n && after < n {
                adj[before].push(after);
                in_degree[after] += 1;
            }
        }

        let mut queue: std::collections::VecDeque<ContourId> = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() == n {
            Some(order)
        } else {
            None // Cycle detected
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_contour(id: ContourId, geom_id: &str, instance: usize, ct: ContourType) -> CutContour {
        CutContour {
            id,
            geometry_id: geom_id.to_string(),
            instance,
            contour_type: ct,
            vertices: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            perimeter: 40.0,
            centroid: (5.0, 5.0),
        }
    }

    #[test]
    fn test_interior_before_exterior() {
        let contours = vec![
            make_contour(0, "part1", 0, ContourType::Exterior),
            make_contour(1, "part1", 0, ContourType::Interior),
            make_contour(2, "part1", 0, ContourType::Interior),
        ];

        let dag = CuttingDag::build(&contours);

        // Interior 1 and 2 must come before Exterior 0
        assert!(dag.is_valid_sequence(&[1, 2, 0]));
        assert!(dag.is_valid_sequence(&[2, 1, 0]));
        assert!(!dag.is_valid_sequence(&[0, 1, 2])); // Exterior first — invalid
    }

    #[test]
    fn test_multiple_parts_no_nesting() {
        // Use non-overlapping contours so nesting rule doesn't fire
        let contours = vec![
            CutContour {
                id: 0,
                geometry_id: "part1".to_string(),
                instance: 0,
                contour_type: ContourType::Exterior,
                vertices: vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                perimeter: 40.0,
                centroid: (5.0, 5.0),
            },
            CutContour {
                id: 1,
                geometry_id: "part2".to_string(),
                instance: 0,
                contour_type: ContourType::Exterior,
                vertices: vec![(50.0, 0.0), (60.0, 0.0), (60.0, 10.0), (50.0, 10.0)],
                perimeter: 40.0,
                centroid: (55.0, 5.0),
            },
        ];

        let dag = CuttingDag::build(&contours);

        // No precedence between different non-overlapping parts
        assert!(dag.is_valid_sequence(&[0, 1]));
        assert!(dag.is_valid_sequence(&[1, 0]));
    }

    #[test]
    fn test_topological_sort() {
        let contours = vec![
            make_contour(0, "part1", 0, ContourType::Exterior),
            make_contour(1, "part1", 0, ContourType::Interior),
        ];

        let dag = CuttingDag::build(&contours);
        let order = dag.topological_sort().expect("DAG should be acyclic");

        assert!(dag.is_valid_sequence(&order));
    }

    #[test]
    fn test_predecessors_and_successors() {
        let contours = vec![
            make_contour(0, "part1", 0, ContourType::Exterior),
            make_contour(1, "part1", 0, ContourType::Interior),
            make_contour(2, "part1", 0, ContourType::Interior),
        ];

        let dag = CuttingDag::build(&contours);

        assert_eq!(dag.predecessors(0).len(), 2); // Both interiors precede exterior
        assert!(dag.successors(1).contains(&0)); // Interior 1 -> Exterior 0
        assert!(dag.successors(2).contains(&0)); // Interior 2 -> Exterior 0
    }

    #[test]
    fn test_violates() {
        let contours = vec![
            make_contour(0, "part1", 0, ContourType::Exterior),
            make_contour(1, "part1", 0, ContourType::Interior),
        ];

        let dag = CuttingDag::build(&contours);

        // Cutting exterior (0) before interior (1) violates
        assert!(dag.violates(0, 1));
        // Cutting interior (1) before exterior (0) is fine
        assert!(!dag.violates(1, 0));
    }

    #[test]
    fn test_nested_parts() {
        // Part2 is inside Part1 (centroid of Part2 is inside Part1's exterior)
        let contours = vec![
            CutContour {
                id: 0,
                geometry_id: "part1".to_string(),
                instance: 0,
                contour_type: ContourType::Exterior,
                vertices: vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                perimeter: 400.0,
                centroid: (50.0, 50.0),
            },
            CutContour {
                id: 1,
                geometry_id: "part2".to_string(),
                instance: 0,
                contour_type: ContourType::Exterior,
                vertices: vec![(20.0, 20.0), (40.0, 20.0), (40.0, 40.0), (20.0, 40.0)],
                perimeter: 80.0,
                centroid: (30.0, 30.0),
            },
        ];

        let dag = CuttingDag::build(&contours);

        // Part2 (inside Part1) must be cut before Part1's exterior
        assert!(dag.is_valid_sequence(&[1, 0]));
        assert!(!dag.is_valid_sequence(&[0, 1]));
    }
}
