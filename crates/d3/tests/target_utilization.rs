//! `target_utilization` stops the searches that read it: once every box is
//! packed at that utilization or more, the search ends and says so. It used to
//! be validated and never read -- no search stopped on it.

use u_nesting_d3::{Boundary3D, Config, Geometry3D, Packer3D, Solver, Strategy};

#[test]
fn a_reached_target_stops_the_search() {
    let geometries = vec![Geometry3D::new("box", 2.0, 2.0, 2.0).with_quantity(4)];
    let boundary = Boundary3D::new(10.0, 10.0, 10.0);
    for strategy in [
        Strategy::GeneticAlgorithm,
        Strategy::Brkga,
        Strategy::SimulatedAnnealing,
    ] {
        let config = Config::default()
            .with_strategy(strategy)
            .with_target_utilization(0.0)
            .with_time_limit(30_000)
            .with_seed(1);
        let result = Packer3D::new(config)
            .solve(&geometries, &boundary)
            .expect("a valid request");
        assert!(result.unplaced.is_empty(), "{strategy:?}: every box fits");
        assert!(
            result.target_reached,
            "{strategy:?}: a target of 0 is met by the first complete packing"
        );
    }
}
