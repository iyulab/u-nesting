//! A box left unpacked says why: it can never go into this container --
//! no allowed orientation fits, or it alone outweighs the container's limit --
//! (`exceeds_boundary`), or the packing ran out of room (`no_room`).

use u_nesting_d3::{
    Boundary3D, Config, Geometry3D, OrientationConstraint, Packer3D, Solver, Strategy,
    UnplacedReason,
};

fn reasons(result: &u_nesting_d3::SolveResult<f64>) -> Vec<(&str, UnplacedReason)> {
    result
        .unplaced_reasons
        .iter()
        .map(|(id, reason)| (id.as_str(), *reason))
        .collect()
}

#[test]
fn a_box_larger_than_the_container_exceeds_it_and_a_full_container_has_no_room() {
    let boxes = [
        Geometry3D::new("big", 50.0, 50.0, 50.0),
        Geometry3D::new("crate", 4.0, 4.0, 4.0).with_quantity(40),
    ];
    let container = Boundary3D::new(10.0, 10.0, 10.0);
    for strategy in [Strategy::BottomLeftFill, Strategy::ExtremePoint] {
        let result = Packer3D::new(Config::default().with_strategy(strategy))
            .solve(&boxes, &container)
            .expect("solved");
        assert_eq!(
            result.unplaced.len(),
            result.unplaced_reasons.len(),
            "{strategy:?}"
        );
        let by_id = reasons(&result);
        assert!(
            by_id.contains(&("big", UnplacedReason::ExceedsBoundary)),
            "{strategy:?}: {by_id:?}"
        );
        assert!(
            by_id.contains(&("crate", UnplacedReason::NoRoom)),
            "{strategy:?}: {by_id:?}"
        );
    }
}

#[test]
fn only_an_allowed_orientation_saves_a_box_and_a_heavy_one_never_fits() {
    let container = Boundary3D::new(10.0, 10.0, 4.0).with_max_mass(100.0);
    let long = |o| Geometry3D::new("plank", 3.0, 3.0, 8.0).with_orientation(o);
    // Lying down it is 8 x 3 x 3 and fits; upright it is 8 tall in a 4-high container.
    let any = Packer3D::new(Config::default())
        .solve(&[long(OrientationConstraint::Any)], &container)
        .expect("solved");
    assert!(any.unplaced.is_empty(), "{:?}", any.unplaced);
    let fixed = Packer3D::new(Config::default())
        .solve(&[long(OrientationConstraint::Fixed)], &container)
        .expect("solved");
    assert_eq!(
        reasons(&fixed),
        vec![("plank", UnplacedReason::ExceedsBoundary)]
    );

    let heavy = Geometry3D::new("anvil", 1.0, 1.0, 1.0).with_mass(500.0);
    let result = Packer3D::new(Config::default())
        .solve(&[heavy], &container)
        .expect("solved");
    assert_eq!(
        reasons(&result),
        vec![("anvil", UnplacedReason::ExceedsBoundary)]
    );
}
