//! Checking a 2D layout against its problem.
//!
//! [`check_layout`] judges a set of placements -- a solve's own, or one from
//! anywhere else -- by the rules the nester places under: every piece on its
//! sheet within the boundary and its margin, no two pieces on a sheet
//! overlapping or closer than the spacing, each placement at a rotation its
//! geometry allows and flipped only where flipping is allowed, each copy placed
//! at most once, and the geometries left unplaced reported as such.

use std::collections::{HashMap, HashSet};

use crate::boundary::Boundary2D;
use crate::geometry::Geometry2D;
use crate::polygon_ops::{mirror_polygon, ring_distance};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::Placement;

/// One way a layout breaks its problem. `placement` and `other` are indices
/// into the placements checked; `id` is a geometry id.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutViolation {
    /// The placement names a geometry the problem does not have.
    UnknownGeometry {
        /// Placement index.
        placement: usize,
        /// The id named.
        id: String,
    },
    /// The placement's copy index is not below the geometry's quantity.
    InstanceOutOfRange {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
        /// The copy index given.
        instance: usize,
        /// The geometry's quantity.
        quantity: usize,
    },
    /// A copy of a geometry is placed a second time.
    InstanceRepeated {
        /// Placement index of the second placement.
        placement: usize,
        /// The geometry.
        id: String,
        /// The copy index.
        instance: usize,
    },
    /// The placement's rotation is not one its geometry allows.
    RotationNotAllowed {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
        /// The rotation given, in radians.
        rotation: f64,
    },
    /// The placement is flipped but its geometry does not allow flipping.
    FlipNotAllowed {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
    },
    /// The placement is on a sheet other than the first in a single-sheet
    /// layout.
    SheetNotAllowed {
        /// Placement index.
        placement: usize,
        /// The sheet index given.
        sheet: usize,
    },
    /// The piece leaves the boundary, or comes closer to its edge than the
    /// margin.
    OutsideBoundary {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
    },
    /// Two pieces on one sheet overlap.
    Overlap {
        /// Placement index of the first piece.
        placement: usize,
        /// Placement index of the second piece.
        other: usize,
    },
    /// Two pieces on one sheet are closer than the spacing.
    TooClose {
        /// Placement index of the first piece.
        placement: usize,
        /// Placement index of the second piece.
        other: usize,
        /// Their distance.
        distance: f64,
        /// The spacing required.
        spacing: f64,
    },
    /// Copies of the geometry are left unplaced, but it is not reported
    /// unplaced.
    MissingFromUnplaced {
        /// The geometry.
        id: String,
    },
    /// The geometry is reported unplaced but every copy is placed, or it is
    /// not a geometry of the problem.
    WronglyUnplaced {
        /// The id reported.
        id: String,
    },
}

/// Checks the layout of a `solve_2d` response against the request it
/// answers -- the one pipeline every binding calls. The request is read as
/// `solve_2d` reads it (the same refusals); a response that is itself a
/// refusal (`success: false`) is refused, there being no layout to check.
#[cfg(feature = "serde")]
pub fn check_2d(
    request: u_nesting_core::api_types::Check2DRequest,
) -> u_nesting_core::api_types::Check2DResponse {
    use u_nesting_core::api_types::{Check2DResponse, ConfigRequest, LayoutViolationResponse as R};
    use u_nesting_core::solver::Config;
    use u_nesting_core::Error;

    let solved = request.solve_result;
    if !solved.success {
        return Check2DResponse::refused(&Error::invalid_option(
            "solve_result",
            "the solve result is a refusal (success: false); give a successful one to check"
                .to_string(),
        ));
    }
    let problem = request.problem;
    let read = || -> u_nesting_core::Result<(Vec<Geometry2D>, Boundary2D, LayoutRules)> {
        let geometries = Geometry2D::from_requests(problem.geometries)?;
        let boundary = Boundary2D::from_request(problem.boundary)?;
        let multi_sheet = problem
            .config
            .as_ref()
            .and_then(|c| c.multi_sheet)
            .unwrap_or(false);
        let config = problem
            .config
            .as_ref()
            .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config)?;
        crate::nester::Nester2D::new(config.clone()).validate(&geometries, &boundary)?;
        Ok((
            geometries,
            boundary,
            LayoutRules {
                spacing: config.spacing,
                margin: config.margin,
                multi_sheet,
            },
        ))
    };
    let (geometries, boundary, rules) = match read() {
        Ok(read) => read,
        Err(e) => return Check2DResponse::refused(&e),
    };
    let placements: Vec<Placement<f64>> = solved
        .placements
        .iter()
        .map(|p| Placement {
            geometry_id: p.geometry_id.clone(),
            instance: p.instance,
            position: vec![p.x, p.y],
            rotation: vec![p.rotation.to_radians()],
            boundary_index: p.sheet_index,
            mirrored: p.flipped,
            rotation_index: None,
        })
        .collect();
    let violations = check_layout(
        &geometries,
        &boundary,
        &rules,
        &placements,
        &solved.unplaced,
    )
    .into_iter()
    .map(|v| match v {
        LayoutViolation::UnknownGeometry { placement, id } => R::UnknownGeometry { placement, id },
        LayoutViolation::InstanceOutOfRange {
            placement,
            id,
            instance,
            quantity,
        } => R::InstanceOutOfRange {
            placement,
            id,
            instance,
            quantity,
        },
        LayoutViolation::InstanceRepeated {
            placement,
            id,
            instance,
        } => R::InstanceRepeated {
            placement,
            id,
            instance,
        },
        LayoutViolation::RotationNotAllowed {
            placement,
            id,
            rotation,
        } => R::RotationNotAllowed {
            placement,
            id,
            rotation: rotation.to_degrees(),
        },
        LayoutViolation::FlipNotAllowed { placement, id } => R::FlipNotAllowed { placement, id },
        LayoutViolation::SheetNotAllowed { placement, sheet } => {
            R::SheetNotAllowed { placement, sheet }
        }
        LayoutViolation::OutsideBoundary { placement, id } => R::OutsideBoundary { placement, id },
        LayoutViolation::Overlap { placement, other } => R::Overlap { placement, other },
        LayoutViolation::TooClose {
            placement,
            other,
            distance,
            spacing,
        } => R::TooClose {
            placement,
            other,
            distance,
            spacing,
        },
        LayoutViolation::MissingFromUnplaced { id } => R::MissingFromUnplaced { id },
        LayoutViolation::WronglyUnplaced { id } => R::WronglyUnplaced { id },
    })
    .collect();
    Check2DResponse::checked(violations)
}

/// The spacing and margin a layout keeps, and whether it may use more than
/// one sheet.
#[derive(Debug, Clone, Copy, Default)]
pub struct LayoutRules {
    /// Least distance between two pieces on a sheet.
    pub spacing: f64,
    /// Least distance between a piece and the boundary's edge.
    pub margin: f64,
    /// Whether pieces may go on sheets after the first (each sheet is the
    /// boundary, coordinates sheet-local).
    pub multi_sheet: bool,
}

/// A piece as placed: its exterior and holes on the sheet.
struct Placed {
    rings: Vec<Vec<(f64, f64)>>,
    min: (f64, f64),
    max: (f64, f64),
    sheet: usize,
}

/// Every way `placements` (with `unplaced`, the ids reported unplaced) break
/// the problem `geometries` on `boundary` under `rules`. Empty means the layout
/// keeps every rule.
///
/// Placements follow the nester's convention: each piece is mirrored (if
/// flipped), rotated by its rotation in radians about its own origin, then
/// translated to its position.
///
/// # Complexity
/// O(p²·v²) for p placements of at most v vertices in the worst case; pairs
/// whose padded bounding boxes are apart are skipped in O(1).
pub fn check_layout(
    geometries: &[Geometry2D],
    boundary: &Boundary2D,
    rules: &LayoutRules,
    placements: &[Placement<f64>],
    unplaced: &[String],
) -> Vec<LayoutViolation> {
    let mut out = Vec::new();
    let by_id: HashMap<&str, &Geometry2D> =
        geometries.iter().map(|g| (g.id().as_str(), g)).collect();
    let (b_min, b_max) = boundary.aabb();
    let scale = [b_min[0], b_min[1], b_max[0], b_max[1]]
        .iter()
        .filter(|v| v.is_finite() && v.abs() < 1e12)
        .fold(1.0_f64, |m, v| m.max(v.abs()));
    let tolerance = 1e-9 * scale + 1e-9;

    let mut seen: HashSet<(&str, usize)> = HashSet::new();
    let mut placed_count: HashMap<&str, usize> = HashMap::new();
    let mut pieces: Vec<Option<Placed>> = Vec::with_capacity(placements.len());
    for (i, p) in placements.iter().enumerate() {
        let Some(geom) = by_id.get(p.geometry_id.as_str()) else {
            out.push(LayoutViolation::UnknownGeometry {
                placement: i,
                id: p.geometry_id.clone(),
            });
            pieces.push(None);
            continue;
        };
        let id = geom.id().as_str();
        if p.instance >= geom.quantity() {
            out.push(LayoutViolation::InstanceOutOfRange {
                placement: i,
                id: id.to_string(),
                instance: p.instance,
                quantity: geom.quantity(),
            });
        } else if !seen.insert((id, p.instance)) {
            out.push(LayoutViolation::InstanceRepeated {
                placement: i,
                id: id.to_string(),
                instance: p.instance,
            });
        } else {
            *placed_count.entry(id).or_default() += 1;
        }
        let rotation = p.rotation.first().copied().unwrap_or(0.0);
        if !geom.rotations().iter().any(|&a| same_angle(a, rotation)) {
            out.push(LayoutViolation::RotationNotAllowed {
                placement: i,
                id: id.to_string(),
                rotation,
            });
        }
        if p.mirrored && !geom.allow_flip() {
            out.push(LayoutViolation::FlipNotAllowed {
                placement: i,
                id: id.to_string(),
            });
        }
        if p.boundary_index > 0 && !rules.multi_sheet {
            out.push(LayoutViolation::SheetNotAllowed {
                placement: i,
                sheet: p.boundary_index,
            });
        }
        if !crate::is_placement_within_bounds(p, geom, boundary, rules.margin, tolerance) {
            out.push(LayoutViolation::OutsideBoundary {
                placement: i,
                id: id.to_string(),
            });
        }
        pieces.push(Some(place(geom, p)));
    }

    for i in 0..pieces.len() {
        let Some(a) = &pieces[i] else { continue };
        for (j, b) in pieces.iter().enumerate().skip(i + 1) {
            let Some(b) = b else { continue };
            if a.sheet != b.sheet || apart(a, b, rules.spacing + tolerance) {
                continue;
            }
            match separation(a, b) {
                None => out.push(LayoutViolation::Overlap {
                    placement: i,
                    other: j,
                }),
                Some(distance) if distance < rules.spacing - tolerance => {
                    out.push(LayoutViolation::TooClose {
                        placement: i,
                        other: j,
                        distance,
                        spacing: rules.spacing,
                    })
                }
                Some(_) => {}
            }
        }
    }

    let reported: HashSet<&str> = unplaced.iter().map(String::as_str).collect();
    for g in geometries {
        let id = g.id().as_str();
        let short = placed_count.get(id).copied().unwrap_or(0) < g.quantity();
        if short && !reported.contains(id) {
            out.push(LayoutViolation::MissingFromUnplaced { id: id.to_string() });
        }
    }
    let mut listed = HashSet::new();
    for id in unplaced {
        let complete = by_id
            .get(id.as_str())
            .is_some_and(|g| placed_count.get(id.as_str()).copied().unwrap_or(0) >= g.quantity());
        if !by_id.contains_key(id.as_str()) || complete || !listed.insert(id.as_str()) {
            out.push(LayoutViolation::WronglyUnplaced { id: id.clone() });
        }
    }
    out
}

/// Whether two angles in radians name the same orientation.
fn same_angle(a: f64, b: f64) -> bool {
    let d = (a - b).rem_euclid(std::f64::consts::TAU);
    d < 1e-6 || std::f64::consts::TAU - d < 1e-6
}

/// The piece's rings on the sheet: mirrored if flipped, rotated, translated.
fn place(geom: &Geometry2D, p: &Placement<f64>) -> Placed {
    let x = p.position.first().copied().unwrap_or(0.0);
    let y = p.position.get(1).copied().unwrap_or(0.0);
    let rotation = p.rotation.first().copied().unwrap_or(0.0);
    let (sin, cos) = rotation.sin_cos();
    let transform = |ring: &[(f64, f64)]| -> Vec<(f64, f64)> {
        let ring = if p.mirrored {
            mirror_polygon(ring)
        } else {
            ring.to_vec()
        };
        ring.into_iter()
            .map(|(vx, vy)| (x + vx * cos - vy * sin, y + vx * sin + vy * cos))
            .collect()
    };
    let rings: Vec<Vec<(f64, f64)>> = std::iter::once(geom.exterior())
        .chain(geom.holes().iter().map(Vec::as_slice))
        .map(transform)
        .collect();
    let (mut min, mut max) = (
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for &(vx, vy) in &rings[0] {
        min = (min.0.min(vx), min.1.min(vy));
        max = (max.0.max(vx), max.1.max(vy));
    }
    Placed {
        rings,
        min,
        max,
        sheet: p.boundary_index,
    }
}

/// Whether the bounding boxes of two pieces are more than `gap` apart.
fn apart(a: &Placed, b: &Placed, gap: f64) -> bool {
    a.max.0 + gap < b.min.0
        || b.max.0 + gap < a.min.0
        || a.max.1 + gap < b.min.1
        || b.max.1 + gap < a.min.1
}

/// The distance between two pieces' material, or `None` when they overlap.
///
/// They overlap when their material (exterior minus holes) shares area --
/// decided by the area of the intersection, so pieces that only touch (along
/// an edge or at a point) do not. Otherwise the distance between the pieces is
/// the least distance between any ring of one and any ring of the other: a
/// piece beside another is measured to its exterior, a piece in another's hole
/// to that hole's edge.
fn separation(a: &Placed, b: &Placed) -> Option<f64> {
    use i_overlay::core::fill_rule::FillRule;
    use i_overlay::core::overlay_rule::OverlayRule;
    use i_overlay::float::single::SingleFloatOverlay;

    let shape = |p: &Placed| -> Vec<Vec<[f64; 2]>> {
        p.rings
            .iter()
            .map(|r| r.iter().map(|&(x, y)| [x, y]).collect())
            .collect()
    };
    let area = |rings: &[Vec<(f64, f64)>]| -> f64 {
        rings
            .iter()
            .map(|r| crate::polygon_ops::signed_area2(r).abs() / 2.0)
            .fold(0.0, |acc, x| if acc == 0.0 { x } else { acc - x })
    };
    let common: f64 = shape(a)
        .overlay(&shape(b), OverlayRule::Intersect, FillRule::EvenOdd)
        .iter()
        .flatten()
        .map(|contour| {
            let pts: Vec<(f64, f64)> = contour.iter().map(|p| (p[0], p[1])).collect();
            crate::polygon_ops::signed_area2(&pts) / 2.0
        })
        .sum::<f64>()
        .abs();
    // Snapping to the overlay's integer grid leaves at most slivers of this order.
    if common > area(&a.rings).min(area(&b.rings)) * 1e-9 {
        return None;
    }
    Some(
        a.rings
            .iter()
            .flat_map(|ra| b.rings.iter().map(move |rb| ring_distance(ra, rb)))
            .fold(f64::INFINITY, f64::min),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(id: &str, side: f64, quantity: usize) -> Geometry2D {
        Geometry2D::rectangle(id, side, side).with_quantity(quantity)
    }

    fn at(id: &str, instance: usize, x: f64, y: f64) -> Placement<f64> {
        Placement::new_2d(id.to_string(), instance, x, y, 0.0)
    }

    fn rules(spacing: f64, margin: f64) -> LayoutRules {
        LayoutRules {
            spacing,
            margin,
            multi_sheet: false,
        }
    }

    /// Every strategy's own layout keeps every rule -- polygons that turn and
    /// flip, a frame with a hole, spacing and margin, one sheet and several.
    /// The checker shares only the boundary-containment test with the nester.
    #[test]
    fn every_strategy_s_layout_passes_the_check() {
        use crate::nester::Nester2D;
        use u_nesting_core::solver::{Config, Solver, Strategy};

        let geometries = vec![
            Geometry2D::new("l")
                .with_polygon(vec![
                    (0.0, 0.0),
                    (20.0, 0.0),
                    (20.0, 6.0),
                    (6.0, 6.0),
                    (6.0, 14.0),
                    (0.0, 14.0),
                ])
                .with_quantity(3)
                .with_rotations_deg(vec![0.0, 90.0, 180.0, 270.0])
                .with_flip(true),
            Geometry2D::new("tri")
                .with_polygon(vec![(0.0, 0.0), (12.0, 0.0), (0.0, 9.0)])
                .with_quantity(3)
                .with_rotations_deg(vec![0.0, 180.0]),
            Geometry2D::rectangle("frame", 24.0, 24.0).with_hole(vec![
                (6.0, 6.0),
                (18.0, 6.0),
                (18.0, 18.0),
                (6.0, 18.0),
            ]),
            Geometry2D::rectangle("chip", 4.0, 3.0).with_quantity(4),
        ];
        let boundary = Boundary2D::rectangle(60.0, 45.0);
        for strategy in Strategy::offered(u_nesting_core::solver::Dimension::TwoD) {
            let mut config = Config::default()
                .with_strategy(strategy)
                .with_spacing(1.0)
                .with_margin(0.5)
                .with_time_limit(2_000);
            if strategy.reads_setting(u_nesting_core::solver::Dimension::TwoD, "population_size") {
                config = config.with_population_size(6).with_max_generations(4);
            }
            if strategy.reads_setting(u_nesting_core::solver::Dimension::TwoD, "seed") {
                config = config.with_seed(7);
            }
            let rules = LayoutRules {
                spacing: 1.0,
                margin: 0.5,
                multi_sheet: false,
            };
            let nester = Nester2D::new(config.clone());
            let result = nester.solve(&geometries, &boundary).expect("solved");
            let v = check_layout(
                &geometries,
                &boundary,
                &rules,
                &result.placements,
                &result.unplaced,
            );
            assert!(v.is_empty(), "{strategy:?}: {v:?}");

            let small = Boundary2D::rectangle(40.0, 30.0);
            let mut multi = nester
                .solve_multi_strip(&geometries, &small)
                .expect("solved");
            let (b_min, b_max) = small.aabb();
            multi.to_boundary_local(b_max[0] - b_min[0]);
            let multi_rules = LayoutRules {
                multi_sheet: true,
                ..rules
            };
            let v = check_layout(
                &geometries,
                &small,
                &multi_rules,
                &multi.placements,
                &multi.unplaced,
            );
            assert!(v.is_empty(), "{strategy:?} multi-sheet: {v:?}");
        }
    }

    #[test]
    fn a_valid_layout_has_no_violations() {
        let g = [square("a", 10.0, 2)];
        let b = Boundary2D::rectangle(30.0, 20.0);
        let p = [at("a", 0, 1.0, 1.0), at("a", 1, 13.0, 1.0)];
        assert!(check_layout(&g, &b, &rules(2.0, 1.0), &p, &[]).is_empty());
        // Abutting is fine at spacing 0.
        let touching = [at("a", 0, 0.0, 0.0), at("a", 1, 10.0, 0.0)];
        assert!(check_layout(&g, &b, &rules(0.0, 0.0), &touching, &[]).is_empty());
    }

    #[test]
    fn overlap_spacing_and_the_boundary_are_checked() {
        let g = [square("a", 10.0, 3)];
        let b = Boundary2D::rectangle(25.0, 20.0);
        // 0 and 1 overlap by 5; 1 and 2 are 1 apart (spacing 2); 2 crosses the
        // right edge (16 + 10 > 25).
        let p = [
            at("a", 0, 0.0, 0.0),
            at("a", 1, 5.0, 0.0),
            at("a", 2, 16.0, 0.0),
        ];
        let v = check_layout(&g, &b, &rules(2.0, 0.0), &p, &[]);
        assert!(
            v.contains(&LayoutViolation::Overlap {
                placement: 0,
                other: 1
            }),
            "{v:?}"
        );
        assert!(
            v.contains(&LayoutViolation::TooClose {
                placement: 1,
                other: 2,
                distance: 1.0,
                spacing: 2.0
            }),
            "{v:?}"
        );
        assert!(
            v.contains(&LayoutViolation::OutsideBoundary {
                placement: 2,
                id: "a".into()
            }),
            "{v:?}"
        );
        // The same square twice at one spot: no edge properly crosses, the
        // interior point decides.
        let same = [at("a", 0, 0.0, 0.0), at("a", 1, 0.0, 0.0)];
        assert_eq!(
            check_layout(&g, &b, &rules(0.0, 0.0), &same, &["a".into()]),
            [LayoutViolation::Overlap {
                placement: 0,
                other: 1
            }]
        );
    }

    #[test]
    fn a_piece_in_a_hole_is_judged_by_the_hole() {
        let frame = Geometry2D::rectangle("frame", 20.0, 20.0).with_hole(vec![
            (5.0, 5.0),
            (15.0, 5.0),
            (15.0, 15.0),
            (5.0, 15.0),
        ]);
        let small = square("small", 4.0, 1);
        let g = [frame, small];
        let b = Boundary2D::rectangle(30.0, 30.0);
        let in_hole = [at("frame", 0, 0.0, 0.0), at("small", 0, 8.0, 8.0)];
        assert!(check_layout(&g, &b, &rules(2.0, 0.0), &in_hole, &[]).is_empty());
        // 1 from the hole's edge with spacing 2; then in the frame's material.
        let near = [at("frame", 0, 0.0, 0.0), at("small", 0, 6.0, 8.0)];
        assert!(matches!(
            check_layout(&g, &b, &rules(2.0, 0.0), &near, &[]).as_slice(),
            [LayoutViolation::TooClose { distance, .. }] if (*distance - 1.0).abs() < 1e-9
        ));
        let on_material = [at("frame", 0, 0.0, 0.0), at("small", 0, 0.5, 0.5)];
        assert_eq!(
            check_layout(&g, &b, &rules(0.0, 0.0), &on_material, &[]),
            [LayoutViolation::Overlap {
                placement: 0,
                other: 1
            }]
        );
    }

    #[test]
    fn copies_rotations_flips_sheets_and_unplaced_are_checked() {
        let g = [square("a", 5.0, 2), square("b", 5.0, 1)];
        let b = Boundary2D::rectangle(50.0, 50.0);
        let mut turned = at("a", 0, 10.0, 10.0);
        turned.rotation = vec![0.3];
        let mut flipped = at("b", 0, 30.0, 30.0);
        flipped.mirrored = true;
        let mut second_sheet = at("a", 1, 40.0, 0.0);
        second_sheet.boundary_index = 1;
        let p = [
            turned,
            at("a", 0, 20.0, 0.0),
            at("a", 2, 0.0, 20.0),
            at("x", 0, 0.0, 40.0),
            flipped,
            second_sheet,
        ];
        let v = check_layout(&g, &b, &rules(0.0, 0.0), &p, &["b".into(), "zz".into()]);
        assert_eq!(
            v,
            [
                LayoutViolation::RotationNotAllowed {
                    placement: 0,
                    id: "a".into(),
                    rotation: 0.3
                },
                LayoutViolation::InstanceRepeated {
                    placement: 1,
                    id: "a".into(),
                    instance: 0
                },
                LayoutViolation::InstanceOutOfRange {
                    placement: 2,
                    id: "a".into(),
                    instance: 2,
                    quantity: 2
                },
                LayoutViolation::UnknownGeometry {
                    placement: 3,
                    id: "x".into()
                },
                LayoutViolation::FlipNotAllowed {
                    placement: 4,
                    id: "b".into()
                },
                LayoutViolation::SheetNotAllowed {
                    placement: 5,
                    sheet: 1
                },
                LayoutViolation::WronglyUnplaced { id: "b".into() },
                LayoutViolation::WronglyUnplaced { id: "zz".into() },
            ]
        );
        let short = [at("a", 0, 0.0, 0.0)];
        assert_eq!(
            check_layout(&g, &b, &rules(0.0, 0.0), &short, &[]),
            [
                LayoutViolation::MissingFromUnplaced { id: "a".into() },
                LayoutViolation::MissingFromUnplaced { id: "b".into() },
            ]
        );
    }
}
