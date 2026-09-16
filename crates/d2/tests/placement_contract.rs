//! The placement contract every 2D strategy shares, measured on the output:
//! `spacing` is the minimum distance between any two placed pieces, and
//! `margin` is the minimum distance from any piece to the boundary edge.
//!
//! Overlap-only checks cannot see either — a layout with half the requested
//! clearance still has zero overlap — so these tests measure the distances.

use std::collections::HashMap;

use proptest::prelude::*;
use u_nesting_d2::{
    Boundary2D, Config, Geometry, Geometry2D, Nester2D, SolveResult, Solver, Strategy,
};

/// Every strategy compiled without the `milp` feature.
const STRATEGIES: [Strategy; 7] = [
    Strategy::BottomLeftFill,
    Strategy::NfpGuided,
    Strategy::GeneticAlgorithm,
    Strategy::Brkga,
    Strategy::SimulatedAnnealing,
    Strategy::Gdrr,
    Strategy::Alns,
];

/// Clearances are computed as offsets that may exceed the request by the arc
/// allowance (0.12 %), and never fall short of it beyond float noise.
const SHORTFALL_TOL: f64 = 1e-6;
const OVERSHOOT: f64 = 1.002;

type Point = (f64, f64);
type Edge = (Point, Point);

/// True when a measured clearance meets `requested` without exceeding the
/// arc allowance.
fn delivers(measured: f64, requested: f64) -> bool {
    (requested - SHORTFALL_TOL..=requested * OVERSHOOT).contains(&measured)
}

fn config(strategy: Strategy) -> Config {
    Config::default()
        .with_strategy(strategy)
        .with_time_limit(1500)
        .with_seed(7)
}

fn world_polygons(pieces: &[Geometry2D], result: &SolveResult<f64>) -> Vec<Vec<(f64, f64)>> {
    let by_id: HashMap<&str, &Geometry2D> = pieces.iter().map(|g| (g.id().as_str(), g)).collect();
    result
        .placements
        .iter()
        .map(|p| {
            let g = by_id[p.geometry_id.as_str()];
            let rot = p.rotation.first().copied().unwrap_or(0.0);
            let (c, s) = (rot.cos(), rot.sin());
            g.exterior()
                .iter()
                .map(|&(x, y)| if p.mirrored { (-x, y) } else { (x, y) })
                .map(|(x, y)| (p.position[0] + x * c - y * s, p.position[1] + x * s + y * c))
                .collect()
        })
        .collect()
}

fn point_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    };
    ((p.0 - a.0 - t * dx).powi(2) + (p.1 - a.1 - t * dy).powi(2)).sqrt()
}

fn segments_cross(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let orient = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
        (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0)
    };
    let (o1, o2, o3, o4) = (
        orient(a, b, c),
        orient(a, b, d),
        orient(c, d, a),
        orient(c, d, b),
    );
    o1 * o2 < 0.0 && o3 * o4 < 0.0
}

fn point_inside(p: (f64, f64), ring: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let mut j = ring.len() - 1;
    for i in 0..ring.len() {
        let ((xi, yi), (xj, yj)) = (ring[i], ring[j]);
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Distance between two simple polygons; 0 when they overlap.
fn polygon_distance(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let edges =
        |r: &[Point]| -> Vec<Edge> { (0..r.len()).map(|i| (r[i], r[(i + 1) % r.len()])).collect() };
    let (ea, eb) = (edges(a), edges(b));
    if ea
        .iter()
        .any(|&(p, q)| eb.iter().any(|&(r, s)| segments_cross(p, q, r, s)))
        || point_inside(a[0], b)
        || point_inside(b[0], a)
    {
        return 0.0;
    }
    let one_way = |pts: &[Point], es: &[Edge]| {
        pts.iter()
            .flat_map(|&p| {
                es.iter()
                    .map(move |&(q, r)| point_segment_distance(p, q, r))
            })
            .fold(f64::INFINITY, f64::min)
    };
    one_way(a, &eb).min(one_way(b, &ea))
}

/// Smallest distance between any two placed pieces (∞ with fewer than two).
fn min_gap(polys: &[Vec<(f64, f64)>]) -> f64 {
    let mut gap = f64::INFINITY;
    for i in 0..polys.len() {
        for j in i + 1..polys.len() {
            gap = gap.min(polygon_distance(&polys[i], &polys[j]));
        }
    }
    gap
}

/// Smallest distance from any placed vertex to the edge of a `w × h` boundary.
fn min_edge_clearance(polys: &[Vec<(f64, f64)>], w: f64, h: f64) -> f64 {
    polys
        .iter()
        .flatten()
        .map(|&(x, y)| x.min(y).min(w - x).min(h - y))
        .fold(f64::INFINITY, f64::min)
}

/// GitHub #3: two squares must end up exactly `spacing` apart — not 20–30 %
/// closer, and not a sampling step farther.
#[test]
fn two_squares_are_placed_exactly_spacing_apart() {
    let pieces = [Geometry2D::rectangle("A", 300.0, 300.0)
        .with_quantity(2)
        .with_rotations(vec![0.0])];
    let boundary = Boundary2D::rectangle(1000.0, 2000.0);
    for strategy in STRATEGIES {
        for spacing in [50.0, 100.0, 150.0] {
            let result = Nester2D::new(config(strategy).with_spacing(spacing))
                .solve(&pieces, &boundary)
                .unwrap();
            let polys = world_polygons(&pieces, &result);
            assert_eq!(
                polys.len(),
                2,
                "{strategy:?} spacing={spacing}: placed {}",
                polys.len()
            );
            let gap = min_gap(&polys);
            assert!(
                delivers(gap, spacing),
                "{strategy:?} spacing={spacing}: gap {gap}"
            );
            // Spacing is between pieces; with no margin a piece may touch the edge.
            assert!(
                min_edge_clearance(&polys, 1000.0, 2000.0).abs() < SHORTFALL_TOL,
                "{strategy:?} spacing={spacing}: first piece pushed off the corner"
            );
        }
    }
}

/// GitHub #4, first repro: the margin is applied once.
#[test]
fn a_single_piece_sits_at_the_margin_corner() {
    let pieces = [Geometry2D::rectangle("a", 100.0, 100.0).with_rotations(vec![0.0])];
    let boundary = Boundary2D::rectangle(1000.0, 1000.0);
    for strategy in STRATEGIES {
        for margin in [0.0, 50.0, 100.0, 150.0] {
            let result = Nester2D::new(config(strategy).with_margin(margin))
                .solve(&pieces, &boundary)
                .unwrap();
            assert_eq!(result.placements.len(), 1, "{strategy:?} margin={margin}");
            let p = &result.placements[0];
            assert!(
                (p.position[0] - margin).abs() < SHORTFALL_TOL
                    && (p.position[1] - margin).abs() < SHORTFALL_TOL,
                "{strategy:?} margin={margin}: placed at ({}, {})",
                p.position[0],
                p.position[1]
            );
        }
    }
}

/// GitHub #4, second repro: a 400 × 400 sheet with margin 100 leaves exactly
/// room for four 100 × 100 pieces.
#[test]
fn a_tight_margin_still_admits_what_fits() {
    let pieces = [Geometry2D::rectangle("a", 100.0, 100.0)
        .with_quantity(4)
        .with_rotations(vec![0.0])];
    let boundary = Boundary2D::rectangle(400.0, 400.0);
    for strategy in STRATEGIES {
        let result = Nester2D::new(config(strategy).with_margin(100.0))
            .solve(&pieces, &boundary)
            .unwrap();
        assert_eq!(
            result.placements.len(),
            4,
            "{strategy:?}: unplaced {:?}",
            result.unplaced
        );
        let polys = world_polygons(&pieces, &result);
        assert!(min_edge_clearance(&polys, 400.0, 400.0) >= 100.0 - SHORTFALL_TOL);
    }
}

/// The progress-reporting entry point runs its own copy of the NFP loop.
#[test]
fn the_progress_path_honours_spacing_and_margin() {
    let pieces = [Geometry2D::rectangle("A", 300.0, 300.0)
        .with_quantity(2)
        .with_rotations(vec![0.0])];
    let boundary = Boundary2D::rectangle(1000.0, 2000.0);
    let nester = Nester2D::new(
        config(Strategy::NfpGuided)
            .with_spacing(50.0)
            .with_margin(40.0),
    );
    let result = nester
        .solve_with_progress(&pieces, &boundary, Box::new(|_| {}))
        .unwrap();
    let polys = world_polygons(&pieces, &result);
    assert_eq!(polys.len(), 2);
    let gap = min_gap(&polys);
    assert!(delivers(gap, 50.0), "gap {gap}");
    let edge = min_edge_clearance(&polys, 1000.0, 2000.0);
    assert!((edge - 40.0).abs() < SHORTFALL_TOL, "edge clearance {edge}");
}

fn l_piece(id: &str, w: f64, h: f64, t: f64) -> Geometry2D {
    Geometry2D::new(id).with_polygon(vec![(0.0, 0.0), (w, 0.0), (w, t), (t, t), (t, h), (0.0, h)])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Concave and rotated pieces, any spacing and margin: no two pieces closer
    /// than `spacing`, no piece closer than `margin` to the edge.
    #[test]
    fn every_layout_keeps_spacing_and_margin(
        legs in prop::collection::vec((30.0f64..120.0, 30.0f64..120.0, 8.0f64..25.0), 1..4),
        spacing in 0.5f64..25.0,
        margin in 0.0f64..30.0,
        strategy in prop::sample::select(STRATEGIES.to_vec()),
    ) {
        let pieces: Vec<Geometry2D> = legs
            .iter()
            .enumerate()
            .map(|(i, &(w, h, t))| {
                l_piece(&format!("L{i}"), w, h, t)
                    .with_quantity(2)
                    .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0])
            })
            .collect();
        let (w, h) = (600.0, 600.0);
        let result = Nester2D::new(
            config(strategy).with_spacing(spacing).with_margin(margin).with_time_limit(400),
        )
        .solve(&pieces, &Boundary2D::rectangle(w, h))
        .unwrap();
        let polys = world_polygons(&pieces, &result);
        let gap = min_gap(&polys);
        prop_assert!(gap >= spacing - SHORTFALL_TOL, "{strategy:?}: gap {gap} < spacing {spacing}");
        let edge = min_edge_clearance(&polys, w, h);
        prop_assert!(edge >= margin - SHORTFALL_TOL, "{strategy:?}: edge {edge} < margin {margin}");
    }
}

/// Distance from a placed polygon to a boundary ring's edges, or `None` when
/// the polygon is not inside the ring.
fn clearance_inside(poly: &[Point], ring: &[Point]) -> Option<f64> {
    let edges =
        |r: &[Point]| -> Vec<Edge> { (0..r.len()).map(|i| (r[i], r[(i + 1) % r.len()])).collect() };
    let (ep, er) = (edges(poly), edges(ring));
    let crosses = ep
        .iter()
        .any(|&(p, q)| er.iter().any(|&(r, s)| segments_cross(p, q, r, s)));
    if crosses || !point_inside(poly[0], ring) {
        return None;
    }
    let one_way = |pts: &[Point], es: &[Edge]| {
        pts.iter()
            .flat_map(|&p| {
                es.iter()
                    .map(move |&(q, r)| point_segment_distance(p, q, r))
            })
            .fold(f64::INFINITY, f64::min)
    };
    Some(one_way(poly, &er).min(one_way(ring, &ep)))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// A non-rectangular boundary — slanted and concave — with mirrored and
    /// rotated pieces: every piece inside it, at least `margin` from its edges,
    /// and at least `spacing` from every other piece.
    #[test]
    fn a_polygon_boundary_keeps_pieces_inside_by_the_margin(
        legs in prop::collection::vec((30.0f64..90.0, 30.0f64..90.0, 8.0f64..25.0), 1..3),
        spacing in 0.5f64..15.0,
        margin in 0.0f64..25.0,
        concave in any::<bool>(),
        strategy in prop::sample::select(STRATEGIES.to_vec()),
    ) {
        let ring: Vec<Point> = if concave {
            vec![(0.0, 0.0), (400.0, 0.0), (400.0, 150.0), (150.0, 150.0), (150.0, 400.0), (0.0, 400.0)]
        } else {
            vec![(0.0, 0.0), (400.0, 0.0), (0.0, 400.0)]
        };
        let pieces: Vec<Geometry2D> = legs
            .iter()
            .enumerate()
            .map(|(i, &(w, h, t))| {
                l_piece(&format!("L{i}"), w, h, t)
                    .with_quantity(8)
                    .with_rotations_deg(vec![0.0, 90.0])
                    .with_flip(true)
            })
            .collect();
        let result = Nester2D::new(
            config(strategy).with_spacing(spacing).with_margin(margin).with_time_limit(400),
        )
        .solve(&pieces, &Boundary2D::new(ring.clone()))
        .unwrap();
        let polys = world_polygons(&pieces, &result);
        for (i, poly) in polys.iter().enumerate() {
            let clearance = clearance_inside(poly, &ring);
            prop_assert!(clearance.is_some(), "{strategy:?}: piece {i} is outside the boundary");
            let clearance = clearance.unwrap_or_default();
            prop_assert!(clearance >= margin - SHORTFALL_TOL, "{strategy:?}: piece {i} is {clearance} from the edge < margin {margin}");
        }
        let gap = min_gap(&polys);
        prop_assert!(gap >= spacing - SHORTFALL_TOL, "{strategy:?}: gap {gap} < spacing {spacing}");
    }
}

/// A triangular sheet filled until pieces reach the slanted edge: the margin
/// holds along it, not only along the two axis-aligned edges.
#[test]
fn the_margin_holds_along_a_slanted_edge() {
    let ring: Vec<Point> = vec![(0.0, 0.0), (600.0, 0.0), (0.0, 600.0)];
    let pieces = [Geometry2D::rectangle("s", 70.0, 70.0)
        .with_quantity(24)
        .with_rotations(vec![0.0])];
    for strategy in STRATEGIES {
        // The greedy strategies get no time limit, so the count below measures
        // the packing rather than how fast the build is.
        let greedy = matches!(strategy, Strategy::BottomLeftFill | Strategy::NfpGuided);
        let result = Nester2D::new(
            config(strategy)
                .with_spacing(10.0)
                .with_margin(30.0)
                .with_time_limit(if greedy { 0 } else { 1500 }),
        )
        .solve(&pieces, &Boundary2D::new(ring.clone()))
        .unwrap();
        let polys = world_polygons(&pieces, &result);
        if strategy == Strategy::NfpGuided {
            // Rows of five, four, three and two reach the slanted edge by 14 pieces.
            assert!(
                polys.len() >= 14,
                "{strategy:?}: placed only {}",
                polys.len()
            );
        }
        for (i, poly) in polys.iter().enumerate() {
            let clearance = clearance_inside(poly, &ring)
                .unwrap_or_else(|| panic!("{strategy:?}: piece {i} is outside the boundary"));
            assert!(
                clearance >= 30.0 - SHORTFALL_TOL,
                "{strategy:?}: piece {i} is {clearance} from the edge"
            );
        }
        assert!(min_gap(&polys) >= 10.0 - SHORTFALL_TOL, "{strategy:?}");
    }
}

/// A sheet with a hole: the ring around the hole holds 64 squares, and the
/// NFP-based placement finds all of them instead of placing parts over the
/// hole and losing them to the boundary check.
#[test]
fn the_ring_around_a_hole_is_filled() {
    let hole: Vec<Point> = vec![
        (200.0, 200.0),
        (800.0, 200.0),
        (800.0, 800.0),
        (200.0, 800.0),
    ];
    let boundary = Boundary2D::rectangle(1000.0, 1000.0).with_hole(hole);
    let pieces = [Geometry2D::rectangle("s", 100.0, 100.0)
        .with_quantity(80)
        .with_rotations(vec![0.0])];
    // 100 cells of a 10x10 grid less the 6x6 the hole covers. Every strategy
    // has to reach it: a packer that reads only the sheet's outline lays
    // pieces over the hole and the placement filter drops them, so the ring
    // comes back part-empty while the layout looks successful.
    for strategy in STRATEGIES {
        let result = Nester2D::new(config(strategy).with_time_limit(0))
            .solve(&pieces, &boundary)
            .unwrap();
        assert_eq!(
            result.placements.len(),
            64,
            "{strategy:?} filled {} of the 64 cells around the hole",
            result.placements.len()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Every piece stays inside the sheet, outside the hole, at least `margin`
    /// from both, and at least `spacing` from every other piece.
    #[test]
    fn a_hole_keeps_pieces_out_by_the_margin(
        hole in (60.0f64..160.0, 60.0f64..160.0, 60.0f64..180.0, 60.0f64..180.0),
        legs in prop::collection::vec((25.0f64..60.0, 25.0f64..60.0, 8.0f64..20.0), 1..3),
        spacing in 0.5f64..10.0,
        margin in 0.0f64..15.0,
        strategy in prop::sample::select(STRATEGIES.to_vec()),
    ) {
        let (hx, hy, hw, hh) = hole;
        let hole_ring: Vec<Point> = vec![(hx, hy), (hx + hw, hy), (hx + hw, hy + hh), (hx, hy + hh)];
        let (w, h) = (400.0, 400.0);
        let outer: Vec<Point> = vec![(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
        let boundary = Boundary2D::rectangle(w, h).with_hole(hole_ring.clone());
        let pieces: Vec<Geometry2D> = legs
            .iter()
            .enumerate()
            .map(|(i, &(lw, lh, t))| {
                l_piece(&format!("L{i}"), lw, lh, t)
                    .with_quantity(6)
                    .with_rotations_deg(vec![0.0, 90.0])
                    .with_flip(true)
            })
            .collect();
        let result = Nester2D::new(
            config(strategy).with_spacing(spacing).with_margin(margin).with_time_limit(400),
        )
        .solve(&pieces, &boundary)
        .unwrap();
        let polys = world_polygons(&pieces, &result);
        for (i, poly) in polys.iter().enumerate() {
            let to_edge = clearance_inside(poly, &outer);
            prop_assert!(to_edge.is_some(), "{strategy:?}: piece {i} leaves the sheet");
            prop_assert!(to_edge.unwrap_or_default() >= margin - SHORTFALL_TOL, "{strategy:?}: piece {i} too close to the sheet edge");
            // `polygon_distance` is 0 both for touching and for overlapping.
            let hole_gap = polygon_distance(poly, &hole_ring);
            let crosses = (0..poly.len()).any(|a| {
                (0..hole_ring.len()).any(|b| {
                    segments_cross(
                        poly[a],
                        poly[(a + 1) % poly.len()],
                        hole_ring[b],
                        hole_ring[(b + 1) % hole_ring.len()],
                    )
                })
            });
            let overlaps_hole =
                crosses || point_inside(poly[0], &hole_ring) || point_inside(hole_ring[0], poly);
            prop_assert!(
                !overlaps_hole && hole_gap >= margin - SHORTFALL_TOL,
                "{strategy:?}: piece {i} is {hole_gap} from the hole (margin {margin})"
            );
        }
        let gap = min_gap(&polys);
        prop_assert!(gap >= spacing - SHORTFALL_TOL, "{strategy:?}: gap {gap} < spacing {spacing}");
    }
}
