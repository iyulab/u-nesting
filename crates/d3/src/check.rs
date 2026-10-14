//! Checking a 3D packing against its problem.
//!
//! [`check_packing`] judges a set of placements -- a solve's own, or one from
//! anywhere else -- by the rules the packer packs under: every box inside the
//! container and its margin, no two boxes overlapping or closer than the
//! spacing, each box in an orientation its geometry allows, each copy packed at
//! most once, the total mass within the container's limit, every box supported
//! where gravity or stability is required, and the geometries left unpacked
//! reported as such.

use std::collections::{HashMap, HashSet};

use crate::boundary::Boundary3D;
use crate::geometry::Geometry3D;
use crate::stability::{PlacedBox, StabilityAnalyzer, StabilityConstraint};
use u_nesting_core::geom::nalgebra_types::{NaPoint3 as Point3, NaVector3 as Vector3};
use u_nesting_core::geometry::Geometry;
use u_nesting_core::Placement;

/// One way a packing breaks its problem. `placement` and `other` are indices
/// into the placements checked; `id` is a geometry id.
#[derive(Debug, Clone, PartialEq)]
pub enum PackingViolation {
    /// The placement names a geometry the problem does not have.
    UnknownGeometry {
        /// Placement index.
        placement: usize,
        /// The id named.
        id: String,
    },
    /// The copy index is not below the geometry's quantity.
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
    /// A copy is packed a second time.
    InstanceRepeated {
        /// Placement index of the second placement.
        placement: usize,
        /// The geometry.
        id: String,
        /// The copy index.
        instance: usize,
    },
    /// The orientation is not one the geometry allows.
    OrientationNotAllowed {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
        /// The orientation index given.
        orientation: usize,
    },
    /// The placement is in a bin other than the one container.
    BinNotAllowed {
        /// Placement index.
        placement: usize,
        /// The bin index given.
        bin: usize,
    },
    /// The box leaves the container, or comes closer to a wall than the margin.
    OutsideBoundary {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
    },
    /// Two boxes overlap.
    Overlap {
        /// Placement index of the first box.
        placement: usize,
        /// Placement index of the second box.
        other: usize,
    },
    /// Two boxes are closer than the spacing.
    TooClose {
        /// Placement index of the first box.
        placement: usize,
        /// Placement index of the second box.
        other: usize,
        /// Their distance.
        distance: f64,
        /// The spacing required.
        spacing: f64,
    },
    /// The packed boxes weigh more than the container takes.
    MassExceeded {
        /// The packed mass.
        mass: f64,
        /// The container's limit.
        max_mass: f64,
    },
    /// A box is not supported as gravity or stability requires.
    Unsupported {
        /// Placement index.
        placement: usize,
        /// The geometry.
        id: String,
    },
    /// Copies are left unpacked, but the geometry is not reported unpacked.
    MissingFromUnplaced {
        /// The geometry.
        id: String,
    },
    /// The geometry is reported unpacked but every copy is packed, it is
    /// listed twice, or it is not a geometry of the problem.
    WronglyUnplaced {
        /// The id reported.
        id: String,
    },
}

/// Checks the packing of a `solve_3d` response against the request it
/// answers -- the one pipeline every binding calls. The request is read as
/// `solve_3d` reads it (the same refusals); a response that is itself a
/// refusal is refused, there being no packing to check.
#[cfg(feature = "serde")]
pub fn check_3d(
    request: u_nesting_core::api_types::Check3DRequest,
) -> u_nesting_core::api_types::CheckResponse {
    use u_nesting_core::api_types::{CheckResponse, ConfigRequest, LayoutViolationResponse as R};
    use u_nesting_core::solver::Config;
    use u_nesting_core::Error;

    let solved = request.solve_result;
    if !solved.success {
        return CheckResponse::refused(&Error::invalid_option(
            "solve_result",
            "the solve result is a refusal (success: false); give a successful one to check"
                .to_string(),
        ));
    }
    let problem = request.problem;
    let read = || -> u_nesting_core::Result<(Vec<Geometry3D>, Boundary3D, PackingRules)> {
        let geometries = Geometry3D::from_requests(problem.geometries)?;
        let boundary = Boundary3D::from_request(problem.boundary);
        let config = problem
            .config
            .as_ref()
            .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config_3d)?;
        crate::packer::Packer3D::new(config.clone()).validate(&geometries, &boundary)?;
        Ok((
            geometries,
            boundary,
            PackingRules {
                spacing: config.spacing,
                margin: config.margin,
            },
        ))
    };
    let (geometries, boundary, rules) = match read() {
        Ok(read) => read,
        Err(e) => return CheckResponse::refused(&e),
    };
    let by_id: HashMap<&str, &Geometry3D> =
        geometries.iter().map(|g| (g.id().as_str(), g)).collect();
    // An orientation is given by its label; one the geometry does not allow
    // becomes an index past its orientations, which the check reports.
    let placements: Vec<Placement<f64>> = solved
        .placements
        .iter()
        .map(|p| {
            let index = by_id.get(p.geometry_id.as_str()).map(|g| {
                let n = g.allowed_orientations().len();
                (0..n)
                    .find(|&i| g.orientation_label(i) == p.orientation)
                    .unwrap_or(n)
            });
            let mut placement = Placement::new_3d(
                p.geometry_id.clone(),
                p.instance,
                p.x,
                p.y,
                p.z,
                0.0,
                0.0,
                0.0,
            )
            .with_rotation_index(index.unwrap_or(0));
            placement.boundary_index = p.bin_index;
            placement
        })
        .collect();
    let violations = check_packing(
        &geometries,
        &boundary,
        &rules,
        &placements,
        &solved.unplaced,
    )
    .into_iter()
    .map(|v| match v {
        PackingViolation::UnknownGeometry { placement, id } => R::UnknownGeometry { placement, id },
        PackingViolation::InstanceOutOfRange {
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
        PackingViolation::InstanceRepeated {
            placement,
            id,
            instance,
        } => R::InstanceRepeated {
            placement,
            id,
            instance,
        },
        PackingViolation::OrientationNotAllowed { placement, id, .. } => R::OrientationNotAllowed {
            placement,
            id,
            orientation: solved.placements[placement].orientation.clone(),
        },
        PackingViolation::BinNotAllowed { placement, bin } => R::BinNotAllowed { placement, bin },
        PackingViolation::OutsideBoundary { placement, id } => R::OutsideBoundary { placement, id },
        PackingViolation::Overlap { placement, other } => R::Overlap { placement, other },
        PackingViolation::TooClose {
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
        PackingViolation::MassExceeded { mass, max_mass } => R::MassExceeded { mass, max_mass },
        PackingViolation::Unsupported { placement, id } => R::Unsupported { placement, id },
        PackingViolation::MissingFromUnplaced { id } => R::MissingFromUnplaced { id },
        PackingViolation::WronglyUnplaced { id } => R::WronglyUnplaced { id },
    })
    .collect();
    CheckResponse::checked(violations)
}

/// The spacing and margin a packing keeps.
#[derive(Debug, Clone, Copy, Default)]
pub struct PackingRules {
    /// Least distance between two boxes.
    pub spacing: f64,
    /// Least distance between a box and the container's walls; the floor is
    /// at this height.
    pub margin: f64,
}

/// Every way `placements` (with `unplaced`, the ids reported unpacked) break
/// the problem `geometries` in `boundary` under `rules`. Empty means the
/// packing keeps every rule.
///
/// A placement's position is its box's minimum corner and its
/// `rotation_index` the orientation (an index into the geometry's
/// [`Geometry3D::allowed_orientations`]; absent = 0), as the packer reports
/// them. Support is judged with the packer's own analysis: any contact below
/// for gravity, 70 % of the base for stability, the floor at `margin`.
///
/// # Complexity
/// O(p²) box pairs for p placements, plus the support analysis.
pub fn check_packing(
    geometries: &[Geometry3D],
    boundary: &Boundary3D,
    rules: &PackingRules,
    placements: &[Placement<f64>],
    unplaced: &[String],
) -> Vec<PackingViolation> {
    let mut out = Vec::new();
    let by_id: HashMap<&str, &Geometry3D> =
        geometries.iter().map(|g| (g.id().as_str(), g)).collect();
    let size = boundary.dimensions();
    let tolerance = 1e-9 * size.iter().fold(1.0_f64, |m, v| m.max(v.abs())) + 1e-9;

    let mut seen: HashSet<(&str, usize)> = HashSet::new();
    let mut packed: HashMap<&str, usize> = HashMap::new();
    let mut boxes: Vec<Option<(Vector3<f64>, Vector3<f64>)>> = Vec::with_capacity(placements.len());
    let mut mass = 0.0;
    for (i, p) in placements.iter().enumerate() {
        let Some(geom) = by_id.get(p.geometry_id.as_str()) else {
            out.push(PackingViolation::UnknownGeometry {
                placement: i,
                id: p.geometry_id.clone(),
            });
            boxes.push(None);
            continue;
        };
        let id = geom.id().as_str();
        if p.instance >= geom.quantity() {
            out.push(PackingViolation::InstanceOutOfRange {
                placement: i,
                id: id.to_string(),
                instance: p.instance,
                quantity: geom.quantity(),
            });
        } else if !seen.insert((id, p.instance)) {
            out.push(PackingViolation::InstanceRepeated {
                placement: i,
                id: id.to_string(),
                instance: p.instance,
            });
        } else {
            *packed.entry(id).or_default() += 1;
        }
        mass += geom.mass().unwrap_or(0.0);
        let orientation = p.rotation_index.unwrap_or(0);
        if orientation >= geom.allowed_orientations().len() {
            out.push(PackingViolation::OrientationNotAllowed {
                placement: i,
                id: id.to_string(),
                orientation,
            });
        }
        if p.boundary_index > 0 {
            out.push(PackingViolation::BinNotAllowed {
                placement: i,
                bin: p.boundary_index,
            });
        }
        let min = Vector3::new(
            p.position.first().copied().unwrap_or(0.0),
            p.position.get(1).copied().unwrap_or(0.0),
            p.position.get(2).copied().unwrap_or(0.0),
        );
        let max = min + geom.dimensions_for_orientation(orientation);
        let inside = (0..3).all(|a| {
            min[a] >= rules.margin - tolerance && max[a] <= size[a] - rules.margin + tolerance
        });
        if !inside {
            out.push(PackingViolation::OutsideBoundary {
                placement: i,
                id: id.to_string(),
            });
        }
        boxes.push(Some((min, max)));
    }

    for i in 0..boxes.len() {
        let Some((a_min, a_max)) = boxes[i] else {
            continue;
        };
        for (j, b) in boxes.iter().enumerate().skip(i + 1) {
            let Some((b_min, b_max)) = *b else { continue };
            // Per-axis gap (negative = the extents overlap on that axis).
            let gap: Vec<f64> = (0..3)
                .map(|a| (b_min[a] - a_max[a]).max(a_min[a] - b_max[a]))
                .collect();
            if gap.iter().all(|&g| g < -tolerance) {
                out.push(PackingViolation::Overlap {
                    placement: i,
                    other: j,
                });
                continue;
            }
            let distance = gap.iter().map(|g| g.max(0.0).powi(2)).sum::<f64>().sqrt();
            if distance < rules.spacing - tolerance {
                out.push(PackingViolation::TooClose {
                    placement: i,
                    other: j,
                    distance,
                    spacing: rules.spacing,
                });
            }
        }
    }

    if let Some(max_mass) = boundary.max_mass() {
        if mass > max_mass + tolerance {
            out.push(PackingViolation::MassExceeded { mass, max_mass });
        }
    }

    if boundary.has_gravity() || boundary.has_stability() {
        let ratio = if boundary.has_stability() { 0.7 } else { 1e-6 };
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::partial_base(ratio))
            .expect("the support ratios used here, 0.7 and 1e-6, are in range");
        let placed: Vec<(usize, PlacedBox)> = placements
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let geom = by_id.get(p.geometry_id.as_str())?;
                let (min, max) = boxes[i]?;
                let mut b = PlacedBox::new(
                    p.geometry_id.clone(),
                    p.instance,
                    Point3::new(min[0], min[1], min[2]),
                    max - min,
                );
                if let Some(m) = geom.mass() {
                    b = b.with_mass(m);
                }
                Some((i, b))
            })
            .collect();
        let only_boxes: Vec<PlacedBox> = placed.iter().map(|(_, b)| b.clone()).collect();
        let report = analyzer.analyze(&only_boxes, rules.margin);
        let unstable: HashSet<(String, usize)> = report
            .unstable_boxes()
            .iter()
            .map(|r| (r.id.clone(), r.instance))
            .collect();
        for (i, p) in placements.iter().enumerate() {
            if unstable.contains(&(p.geometry_id.clone(), p.instance)) {
                out.push(PackingViolation::Unsupported {
                    placement: i,
                    id: p.geometry_id.clone(),
                });
            }
        }
    }

    let reported: HashSet<&str> = unplaced.iter().map(String::as_str).collect();
    for g in geometries {
        let id = g.id().as_str();
        if packed.get(id).copied().unwrap_or(0) < g.quantity() && !reported.contains(id) {
            out.push(PackingViolation::MissingFromUnplaced { id: id.to_string() });
        }
    }
    let mut listed = HashSet::new();
    for id in unplaced {
        let complete = by_id
            .get(id.as_str())
            .is_some_and(|g| packed.get(id.as_str()).copied().unwrap_or(0) >= g.quantity());
        if !by_id.contains_key(id.as_str()) || complete || !listed.insert(id.as_str()) {
            out.push(PackingViolation::WronglyUnplaced { id: id.clone() });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(id: &str, side: f64, quantity: usize) -> Geometry3D {
        Geometry3D::new(id, side, side, side).with_quantity(quantity)
    }

    fn at(id: &str, instance: usize, x: f64, y: f64, z: f64) -> Placement<f64> {
        Placement::new_3d(id.to_string(), instance, x, y, z, 0.0, 0.0, 0.0)
    }

    fn rules(spacing: f64, margin: f64) -> PackingRules {
        PackingRules { spacing, margin }
    }

    #[test]
    fn a_valid_packing_has_no_violations_and_each_rule_is_checked() {
        let g = [cube("c", 10.0, 3)];
        let b = Boundary3D::new(30.0, 20.0, 20.0);
        let ok = [at("c", 0, 0.0, 0.0, 0.0), at("c", 1, 11.0, 0.0, 0.0)];
        assert!(check_packing(&g, &b, &rules(1.0, 0.0), &ok, &["c".into()]).is_empty());

        // Overlapping, 0.5 apart (spacing 1), and through the far wall
        // (15.5 + 10 > 25).
        let b = Boundary3D::new(25.0, 20.0, 20.0);
        let bad = [
            at("c", 0, 0.0, 0.0, 0.0),
            at("c", 1, 5.0, 0.0, 0.0),
            at("c", 2, 15.5, 0.0, 0.0),
        ];
        let v = check_packing(&g, &b, &rules(1.0, 0.0), &bad, &[]);
        assert!(
            v.contains(&PackingViolation::Overlap {
                placement: 0,
                other: 1
            }),
            "{v:?}"
        );
        assert!(
            v.contains(&PackingViolation::TooClose {
                placement: 1,
                other: 2,
                distance: 0.5,
                spacing: 1.0
            }),
            "{v:?}"
        );
        assert!(
            v.contains(&PackingViolation::OutsideBoundary {
                placement: 2,
                id: "c".into()
            }),
            "{v:?}"
        );
    }

    #[test]
    fn mass_support_orientation_bins_copies_and_unplaced_are_checked() {
        let g = [
            cube("heavy", 10.0, 2).with_mass(60.0),
            Geometry3D::new("flat", 10.0, 10.0, 2.0)
                .with_orientation(crate::geometry::OrientationConstraint::Fixed),
        ];
        let b = Boundary3D::new(40.0, 40.0, 40.0)
            .with_max_mass(100.0)
            .with_gravity(true);
        let mut turned = at("flat", 0, 20.0, 0.0, 0.0);
        turned.rotation_index = Some(3);
        let mut other_bin = at("heavy", 1, 0.0, 20.0, 0.0);
        other_bin.boundary_index = 1;
        let p = [
            at("heavy", 0, 0.0, 0.0, 15.0), // floating
            other_bin,
            turned,
            at("ghost", 0, 30.0, 30.0, 0.0),
        ];
        let v = check_packing(&g, &b, &rules(0.0, 0.0), &p, &["flat".into()]);
        assert_eq!(
            v,
            [
                PackingViolation::BinNotAllowed {
                    placement: 1,
                    bin: 1
                },
                PackingViolation::OrientationNotAllowed {
                    placement: 2,
                    id: "flat".into(),
                    orientation: 3
                },
                PackingViolation::UnknownGeometry {
                    placement: 3,
                    id: "ghost".into()
                },
                PackingViolation::MassExceeded {
                    mass: 120.0,
                    max_mass: 100.0
                },
                PackingViolation::Unsupported {
                    placement: 0,
                    id: "heavy".into()
                },
                PackingViolation::WronglyUnplaced { id: "flat".into() },
            ]
        );
    }

    /// Every strategy's own packing keeps every rule -- turned boxes, mass,
    /// gravity, spacing and margin.
    #[test]
    fn every_strategy_s_packing_passes_the_check() {
        use crate::packer::Packer3D;
        use u_nesting_core::solver::{Config, Dimension, Solver, Strategy};

        let geometries = vec![
            Geometry3D::new("long", 20.0, 6.0, 4.0)
                .with_quantity(3)
                .with_mass(5.0),
            Geometry3D::new("cube", 8.0, 8.0, 8.0)
                .with_quantity(4)
                .with_mass(8.0),
            Geometry3D::new("slab", 15.0, 10.0, 3.0)
                .with_quantity(2)
                .with_orientation(crate::geometry::OrientationConstraint::Upright),
        ];
        let boundary = Boundary3D::new(30.0, 25.0, 20.0)
            .with_max_mass(200.0)
            .with_gravity(true);
        for strategy in Strategy::offered(Dimension::ThreeD) {
            let mut config = Config::default()
                .with_strategy(strategy)
                .with_spacing(0.5)
                .with_margin(0.5)
                .with_time_limit(2_000);
            if strategy.reads_setting(Dimension::ThreeD, "population_size") {
                config = config.with_population_size(6).with_max_generations(4);
            }
            if strategy.reads_setting(Dimension::ThreeD, "seed") {
                config = config.with_seed(5);
            }
            let result = Packer3D::new(config)
                .solve(&geometries, &boundary)
                .expect("packed");
            let v = check_packing(
                &geometries,
                &boundary,
                &rules(0.5, 0.5),
                &result.placements,
                &result.unplaced,
            );
            assert!(v.is_empty(), "{strategy:?}: {v:?}");
        }
    }
}
