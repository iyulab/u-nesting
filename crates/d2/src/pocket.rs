//! Pockets: where a piece fits inside a hole of another piece.
//!
//! A no-fit polygon forbids every position at which the moving piece overlaps the
//! stationary piece's outline, so the stationary piece's holes read as filled. A
//! hole is empty space: the moving piece may sit in it, as long as it lies wholly
//! inside. The positions that qualify, for one hole, are its **pocket**.
//!
//! The moving piece lies inside a hole `H` exactly when it lies inside the convex
//! hull of `H` and overlaps none of the *bays* -- the parts of the hull that are
//! not `H`, one for each stretch of `H`'s outline between two hull vertices:
//!
//! - inside the hull: the hull is convex, so the moving piece is inside when its
//!   vertices are -- an erosion of the hull by the piece's vertices, exact
//!   (the inner-fit erosion the sheet boundary uses). The same erosion by `H` itself would not be:
//!   with a non-convex `H` every vertex can be inside while an edge crosses the
//!   notch between them;
//! - clear of each bay: a bay is a simple polygon, so its no-fit polygon with the
//!   piece comes from the same decomposition any non-convex pair uses.
//!
//! A [`Pocket`] keeps the two parts apart, as polygons, and answers "is this
//! position in the pocket" with the predicates the placement search uses for
//! "inside the inner-fit polygon" and "outside a no-fit polygon", so touching the
//! hole's edge is allowed and overlapping it is not.

use crate::geometry::Geometry2D;
use crate::nfp::{
    bounding_box, compute_minkowski_erosion, compute_nfp_general, ensure_ccw, point_in_polygon,
    point_on_polygon_boundary, rotate_nfp, rotate_polygon, translate_nfp, Nfp,
};
use crate::polygon_ops::{mirror_polygon, signed_area2};

/// The positions of the moving piece's reference point at which the piece lies
/// inside one hole of the stationary piece.
#[derive(Debug, Clone)]
pub struct Pocket {
    /// Positions at which the piece lies inside the hole's convex hull (closed).
    pub fit: Nfp,
    /// Positions at which the piece overlaps a bay of the hole (open): each is the
    /// no-fit polygon of one bay. Empty when the hole is convex.
    pub bays: Vec<Nfp>,
}

impl Pocket {
    /// Whether a piece whose reference point is at `point` lies inside the hole.
    pub fn contains(&self, point: (f64, f64)) -> bool {
        let in_fit = self
            .fit
            .polygons
            .iter()
            .any(|p| point_in_polygon(point, p) || point_on_polygon_boundary(point, p));
        in_fit
            && !self
                .bays
                .iter()
                .flat_map(|bay| bay.polygons.iter())
                .any(|p| strictly_inside(point, p))
    }

    /// The pocket with the stationary piece turned by `angle` about the origin.
    pub fn rotated(&self, angle: f64) -> Pocket {
        Pocket {
            fit: rotate_nfp(&self.fit, angle),
            bays: self.bays.iter().map(|b| rotate_nfp(b, angle)).collect(),
        }
    }

    /// The pocket with the stationary piece moved by `offset`.
    pub fn translated(&self, offset: (f64, f64)) -> Pocket {
        Pocket {
            fit: translate_nfp(&self.fit, offset),
            bays: self.bays.iter().map(|b| translate_nfp(b, offset)).collect(),
        }
    }
}

/// Whether `point` is inside `polygon` and not on its outline -- the test the
/// placement search applies to a no-fit polygon, where touching is allowed:
/// outside the bounding box, or on its edge, is outside (a ray-cast alone puts
/// some outline points in and some out).
fn strictly_inside(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let (min_x, min_y, max_x, max_y) = bounding_box(polygon);
    point.0 > min_x
        && point.0 < max_x
        && point.1 > min_y
        && point.1 < max_y
        && point_in_polygon(point, polygon)
}

/// The pockets `orbiting` has in the holes of `stationary`, one per hole it can
/// fit in -- a hole the piece cannot fit in at all has none. Both pieces are
/// transformed as [`crate::nfp::compute_nfp_mirrored`] transforms them: mirrored
/// first when asked, and the orbiting piece then turned by `rotation`.
///
/// `spacing` is the clearance the piece keeps from the material around the hole:
/// the pocket is that much smaller (the hull is shrunk by `spacing`, the bays'
/// no-fit polygons are grown by it -- as a no-fit polygon is, against a piece).
pub fn compute_pockets(
    stationary: &Geometry2D,
    orbiting: &Geometry2D,
    rotation: f64,
    mirror_stationary: bool,
    mirror_orbiting: bool,
    spacing: f64,
) -> Vec<Pocket> {
    let orb = orbiting.exterior();
    if stationary.holes().is_empty() || orb.len() < 3 {
        return Vec::new();
    }
    let base_orbiting = if mirror_orbiting {
        mirror_polygon(orb)
    } else {
        orb.to_vec()
    };
    let rotated_orbiting = rotate_polygon(&base_orbiting, rotation);

    stationary
        .holes()
        .iter()
        .filter(|hole| hole.len() >= 3)
        .filter_map(|hole| {
            let hole = if mirror_stationary {
                mirror_polygon(hole)
            } else {
                hole.clone()
            };
            pocket_in(&ensure_ccw(&hole), &rotated_orbiting, spacing)
        })
        .collect()
}

/// The pocket of `piece` (already turned) in one hole `hole` (counter-clockwise),
/// or `None` when the piece cannot fit in it.
fn pocket_in(hole: &[(f64, f64)], piece: &[(f64, f64)], spacing: f64) -> Option<Pocket> {
    let hull = hull_indices(hole);
    if hull.len() < 3 {
        return None;
    }
    let hull_ring: Vec<(f64, f64)> = hull.iter().map(|&i| hole[i]).collect();
    // Everything outside the hull is material, so the clearance from it is a
    // smaller hull; the bays, which are material too, grow their no-fit polygons.
    let kept = if spacing > 0.0 {
        crate::polygon_ops::offset_polygon(&hull_ring, -spacing)
            .into_iter()
            .max_by(|a, b| signed_area2(a).abs().total_cmp(&signed_area2(b).abs()))
            .map(|ring| ensure_ccw(&ring))?
    } else {
        hull_ring.clone()
    };
    // No position at all when the hull is too small for the piece.
    let fit = compute_minkowski_erosion(&kept, piece).ok()?;
    if fit.is_empty() {
        return None;
    }

    // The bays: each stretch of the hole's outline between two consecutive hull
    // vertices that runs through other vertices, closed by the hull edge.
    let n = hole.len();
    let mut in_ring_order = hull.clone();
    in_ring_order.sort_unstable();
    let area_floor = 1e-9 * signed_area2(&hull_ring).abs();
    let mut bays = Vec::new();
    for (k, &start) in in_ring_order.iter().enumerate() {
        let end = in_ring_order[(k + 1) % in_ring_order.len()];
        let steps = (end + n - start) % n;
        if steps < 2 {
            continue; // the outline runs along the hull edge itself
        }
        let chain: Vec<(f64, f64)> = (0..=steps).map(|s| hole[(start + s) % n]).collect();
        // The outline may run along the hull edge before it leaves it and again
        // after it returns; those stretches cut nothing out, and left in they
        // would fold the bay's outline back over itself.
        let (a, b) = (chain[0], chain[chain.len() - 1]);
        let reach = (b.0 - a.0).hypot(b.1 - a.1);
        let on_edge = |p: &(f64, f64)| {
            ((b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)).abs()
                <= 1e-9 * reach * reach.max(1.0)
        };
        let leave = chain.iter().take_while(|p| on_edge(p)).count().max(1) - 1;
        let back = chain.iter().rev().take_while(|p| on_edge(p)).count().max(1) - 1;
        if leave + back + 2 > chain.len() {
            continue; // every vertex is on the hull edge: nothing is cut out of the hull
        }
        let bay = &chain[leave..chain.len() - back];
        if signed_area2(bay).abs() <= area_floor {
            continue;
        }
        if let Ok(nfp) = compute_nfp_general(bay, piece) {
            bays.push(crate::placement_utils::offset_nfp(&nfp, spacing));
        }
    }
    Some(Pocket { fit, bays })
}

/// The indices of the vertices of `ring` on its convex hull, counter-clockwise.
/// Vertices on a hull edge but between its ends are left out.
fn hull_indices(ring: &[(f64, f64)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ring.len()).collect();
    order.sort_by(|&a, &b| {
        ring[a]
            .0
            .total_cmp(&ring[b].0)
            .then(ring[a].1.total_cmp(&ring[b].1))
    });
    let cross = |o: usize, a: usize, b: usize| -> f64 {
        (ring[a].0 - ring[o].0) * (ring[b].1 - ring[o].1)
            - (ring[a].1 - ring[o].1) * (ring[b].0 - ring[o].0)
    };
    let mut hull: Vec<usize> = Vec::with_capacity(order.len() * 2);
    for &i in &order {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }
    let lower = hull.len() + 1;
    for &i in order.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], i) <= 0.0 {
            hull.pop();
        }
        hull.push(i);
    }
    hull.pop();
    hull
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polygon_ops::signed_area2;
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::single::SingleFloatOverlay;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
        vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }

    fn rect_piece(w: f64, h: f64) -> Geometry2D {
        Geometry2D::rectangle("piece", w, h)
    }

    /// A frame with `hole` cut out of its middle.
    fn frame(hole: Vec<(f64, f64)>) -> Geometry2D {
        Geometry2D::rectangle("frame", 40.0, 40.0).with_hole(hole)
    }

    /// Whether `piece` placed at `at` lies inside `hole`, by the area of the
    /// intersection: a different route from the pocket's (no no-fit polygon, no
    /// hull), so it is the oracle.
    fn lies_inside(piece: &[(f64, f64)], at: (f64, f64), hole: &[(f64, f64)]) -> bool {
        let to_shape = |ring: Vec<(f64, f64)>| -> Vec<Vec<[f64; 2]>> {
            vec![ring.into_iter().map(|(x, y)| [x, y]).collect()]
        };
        let placed: Vec<(f64, f64)> = piece.iter().map(|&(x, y)| (x + at.0, y + at.1)).collect();
        let common: f64 = to_shape(placed.clone())
            .overlay(
                &to_shape(hole.to_vec()),
                OverlayRule::Intersect,
                FillRule::EvenOdd,
            )
            .iter()
            .flatten()
            .map(|contour| {
                let pts: Vec<(f64, f64)> = contour.iter().map(|p| (p[0], p[1])).collect();
                signed_area2(&pts) / 2.0
            })
            .sum::<f64>()
            .abs();
        common >= signed_area2(&placed).abs() / 2.0 * (1.0 - 1e-6)
    }

    /// Every grid position (off the lattice the shapes' coordinates sit on, so
    /// none touches a hole edge) agrees with the oracle.
    fn assert_pocket_matches_oracle(hole: Vec<(f64, f64)>, w: f64, h: f64) {
        let pieces = frame(hole.clone());
        let piece = rect_piece(w, h);
        let pockets = compute_pockets(&pieces, &piece, 0.0, false, false, 0.0);
        let piece_ring = piece.exterior().to_vec();
        let (mut inside, mut outside) = (0, 0);
        let mut x = -3.137;
        while x < 45.0 {
            let mut y = -2.911;
            while y < 45.0 {
                let expected = lies_inside(&piece_ring, (x, y), &hole);
                let got = pockets.iter().any(|p| p.contains((x, y)));
                assert_eq!(got, expected, "position ({x:.3}, {y:.3})");
                if expected {
                    inside += 1
                } else {
                    outside += 1
                }
                y += 0.713;
            }
            x += 0.713;
        }
        assert!(inside > 0 && outside > 0, "the grid covers both sides");
    }

    #[test]
    fn a_square_hole_takes_a_smaller_square_exactly() {
        assert_pocket_matches_oracle(square(10.0, 10.0, 30.0, 30.0), 8.0, 8.0);
    }

    #[test]
    fn a_notched_hole_is_judged_by_the_notch_not_by_the_vertices() {
        // A C: its two arms are inside, the gap between them is not. A 4 x 15 piece
        // at (24, 9.5) has all four corners inside the C -- two in each arm -- and
        // both its long edges across the gap: the vertices say inside, the hole
        // says no.
        let c = vec![
            (8.0, 8.0),
            (32.0, 8.0),
            (32.0, 16.0),
            (20.0, 16.0),
            (20.0, 24.0),
            (32.0, 24.0),
            (32.0, 32.0),
            (8.0, 32.0),
        ];
        let corners = [(24.0, 9.5), (28.0, 9.5), (24.0, 24.5), (28.0, 24.5)];
        assert!(corners.iter().all(|&q| point_in_polygon(q, &c)));
        let piece = rect_piece(4.0, 15.0);
        assert!(!lies_inside(piece.exterior(), (24.0, 9.5), &c));
        let pockets = compute_pockets(&frame(c.clone()), &piece, 0.0, false, false, 0.0);
        assert!(!pockets.iter().any(|p| p.contains((24.0, 9.5))));
        assert_pocket_matches_oracle(c.clone(), 4.0, 15.0);
        assert_pocket_matches_oracle(c, 6.0, 3.0);
    }

    #[test]
    fn touching_a_hole_s_edge_is_allowed_and_overlapping_it_is_not() {
        let square_hole = frame(square(10.0, 10.0, 30.0, 30.0));
        let piece = rect_piece(8.0, 8.0);
        let pocket = &compute_pockets(&square_hole, &piece, 0.0, false, false, 0.0)[0];
        // Against the hole's left and bottom edges, and its right and top edges.
        assert!(pocket.contains((10.0, 10.0)) && pocket.contains((22.0, 22.0)));
        assert!(!pocket.contains((9.99, 10.0)) && !pocket.contains((22.01, 22.0)));

        // Against the face of a notch: the 4 x 4 piece at x = 16 ends at the notch's
        // edge x = 20; a half unit further it is in the notch.
        let c = vec![
            (8.0, 8.0),
            (32.0, 8.0),
            (32.0, 16.0),
            (20.0, 16.0),
            (20.0, 24.0),
            (32.0, 24.0),
            (32.0, 32.0),
            (8.0, 32.0),
        ];
        let pockets = compute_pockets(&frame(c), &rect_piece(4.0, 4.0), 0.0, false, false, 0.0);
        assert!(pockets.iter().any(|p| p.contains((16.0, 18.0))));
        assert!(!pockets.iter().any(|p| p.contains((16.5, 18.0))));
    }

    #[test]
    fn a_hole_with_two_bays_is_judged_bay_by_bay() {
        // A plus-shaped hole has four bays at the hull's corners.
        let plus = vec![
            (16.0, 8.0),
            (24.0, 8.0),
            (24.0, 16.0),
            (32.0, 16.0),
            (32.0, 24.0),
            (24.0, 24.0),
            (24.0, 32.0),
            (16.0, 32.0),
            (16.0, 24.0),
            (8.0, 24.0),
            (8.0, 16.0),
            (16.0, 16.0),
        ];
        assert_pocket_matches_oracle(plus, 5.0, 4.0);
    }

    #[test]
    fn a_convex_hole_has_no_bays_and_a_piece_too_big_has_no_pocket() {
        let g = frame(square(10.0, 10.0, 30.0, 30.0));
        let pockets = compute_pockets(&g, &rect_piece(8.0, 8.0), 0.0, false, false, 0.0);
        assert_eq!(pockets.len(), 1);
        assert!(pockets[0].bays.is_empty());
        assert!(compute_pockets(&g, &rect_piece(21.0, 5.0), 0.0, false, false, 0.0).is_empty());
        // No holes, no pockets.
        let solid = Geometry2D::rectangle("solid", 40.0, 40.0);
        assert!(compute_pockets(&solid, &rect_piece(2.0, 2.0), 0.0, false, false, 0.0).is_empty());
    }

    #[test]
    fn a_pocket_follows_the_stationary_piece() {
        let g = frame(square(10.0, 10.0, 30.0, 30.0));
        let pocket = &compute_pockets(&g, &rect_piece(8.0, 8.0), 0.0, false, false, 0.0)[0];
        assert!(pocket.contains((15.0, 15.0)));
        let moved = pocket.translated((100.0, 50.0));
        assert!(moved.contains((115.0, 65.0)) && !moved.contains((15.0, 15.0)));
        // A quarter turn about the origin takes (15, 15) to (-15, 15).
        let turned = pocket.rotated(std::f64::consts::FRAC_PI_2);
        assert!(turned.contains((-15.0, 15.0)) && !turned.contains((15.0, 15.0)));
    }

    #[test]
    fn the_search_takes_the_pocket_s_least_corner() {
        use crate::nfp::{
            compute_ifp, compute_nfp_mirrored, find_bottom_left_placement, PackingAxis,
        };

        // A 30 x 30 sheet under a frame (40 x 40) whose hole (6..24) takes a 6 x 5 chip: the
        // frame's no-fit polygon covers every position the sheet allows, so a placement
        // exists only in the pocket, and its least corner is the pocket's own. The
        // sample step is too coarse for the grid fallback to find it by luck.
        let g = frame(square(6.0, 6.0, 24.0, 24.0));
        let chip = rect_piece(6.0, 5.0);
        let sheet = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)];
        let ifp = compute_ifp(&sheet, &chip, 0.0).unwrap();
        let bare = compute_nfp_mirrored(&g, &chip, 0.0, false, false).unwrap();
        assert_eq!(
            find_bottom_left_placement(&ifp, &[&bare], 1000.0, PackingAxis::X),
            None,
            "without the pocket the hole reads as filled"
        );
        let pockets = compute_pockets(&g, &chip, 0.0, false, false, 0.0);
        let carved = bare.with_pockets(pockets);
        assert_eq!(
            find_bottom_left_placement(&ifp, &[&carved], 1000.0, PackingAxis::X),
            Some((6.0, 6.0))
        );
    }

    #[test]
    fn the_overlap_check_lets_a_piece_into_a_hole_only_where_the_hole_is_open() {
        use crate::nfp::{verify_no_overlap_mirrored, PlacedGeometry};

        let open = frame(square(6.0, 6.0, 24.0, 24.0)).with_part_in_part(true);
        let shut = frame(square(6.0, 6.0, 24.0, 24.0));
        let chip = rect_piece(6.0, 5.0);
        let verify = |frame: &Geometry2D, at: (f64, f64)| {
            let placed = [PlacedGeometry::new(frame.clone(), (0.0, 0.0), 0.0)];
            verify_no_overlap_mirrored(&chip, at, 0.0, false, &placed)
        };
        // Inside the hole (touching its left edge): the outlines cross, the material does not.
        assert!(verify(&open, (6.0, 10.0)));
        assert!(!verify(&shut, (6.0, 10.0)));
        // Across the hole's edge: material under the chip either way.
        assert!(!verify(&open, (3.0, 10.0)));
        // Outside the frame altogether (the helper's frame is 40 x 40).
        assert!(verify(&open, (50.0, 10.0)) && verify(&shut, (50.0, 10.0)));
    }

    #[test]
    fn a_second_chip_in_the_hole_is_placed_at_a_crossing_of_the_pocket_and_the_first() {
        use crate::nfp::{
            compute_ifp, compute_nfp_mirrored, find_bottom_left_placement, translate_nfp,
            PackingAxis,
        };

        // The frame's pocket for a 6 x 5 chip is x 6..18, y 6..19. A first chip at (6, 6)
        // forbids x 0..12, y 1..11 to the second; the least point left is where the
        // pocket's left edge meets the first chip's top edge -- no vertex of either.
        let g = frame(square(6.0, 6.0, 24.0, 24.0));
        let chip = rect_piece(6.0, 5.0);
        let sheet = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)];
        let ifp = compute_ifp(&sheet, &chip, 0.0).unwrap();
        let carved = compute_nfp_mirrored(&g, &chip, 0.0, false, false)
            .unwrap()
            .with_pockets(compute_pockets(&g, &chip, 0.0, false, false, 0.0));
        let first = translate_nfp(
            &compute_nfp_mirrored(&chip, &chip, 0.0, false, false).unwrap(),
            (6.0, 6.0),
        );
        assert_eq!(
            find_bottom_left_placement(&ifp, &[&carved, &first], 1000.0, PackingAxis::X),
            Some((6.0, 11.0))
        );
    }

    #[test]
    fn a_turned_frame_carries_its_pockets_with_it() {
        use crate::nfp::{compute_nfp_mirrored, rotate_nfp, translate_nfp};

        let g = frame(square(10.0, 10.0, 30.0, 30.0));
        let chip = rect_piece(8.0, 8.0);
        let carved = compute_nfp_mirrored(&g, &chip, 0.0, false, false)
            .unwrap()
            .with_pockets(compute_pockets(&g, &chip, 0.0, false, false, 0.0));
        assert!(carved.pockets[0].contains((15.0, 15.0)));
        // A quarter turn about the origin, then a move: (15, 15) goes to (-15, 15), then (85, 65).
        let placed = translate_nfp(
            &rotate_nfp(&carved, std::f64::consts::FRAC_PI_2),
            (100.0, 50.0),
        );
        assert!(placed.pockets[0].contains((85.0, 65.0)));
        assert!(!placed.pockets[0].contains((115.0, 65.0)));
    }

    #[test]
    fn a_strategy_that_does_not_place_in_holes_refuses_a_piece_that_asks_for_it() {
        use crate::boundary::Boundary2D;
        use crate::nester::Nester2D;
        use u_nesting_core::solver::{Config, Dimension, Solver, Strategy};

        let geometries = vec![
            rect_piece(5.0, 5.0),
            frame(square(6.0, 6.0, 24.0, 24.0)).with_part_in_part(true),
        ];
        let boundary = Boundary2D::rectangle(100.0, 100.0);
        for strategy in Strategy::offered(Dimension::TwoD) {
            let outcome = Nester2D::new(
                Config::default()
                    .with_strategy(strategy)
                    .with_time_limit(1_000),
            )
            .solve(&geometries, &boundary);
            if strategy.places_in_holes() {
                assert!(outcome.is_ok(), "{strategy:?}: {:?}", outcome.err());
            } else {
                let error = outcome.expect_err("refused");
                assert_eq!(
                    error.parameter(),
                    Some("geometries[1].allow_part_in_part"),
                    "{strategy:?}"
                );
            }
        }
        assert!(!Strategy::BottomLeftFill.places_in_holes());
    }

    #[test]
    fn the_hull_walk_finds_the_vertices_of_a_notched_ring() {
        let c = vec![
            (8.0, 8.0),
            (32.0, 8.0),
            (32.0, 16.0),
            (20.0, 16.0),
            (20.0, 24.0),
            (32.0, 24.0),
            (32.0, 32.0),
            (8.0, 32.0),
        ];
        let mut hull = hull_indices(&c);
        hull.sort_unstable();
        // The four corners of the box; the notch's vertices are not on the hull.
        assert_eq!(hull, vec![0, 1, 6, 7]);
    }
}
