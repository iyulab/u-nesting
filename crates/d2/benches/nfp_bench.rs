//! Benchmarks for 2D nesting operations.
//!
//! Measures NFP computation, geometry creation, and solver performance
//! at various scales.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use u_nesting_core::solver::Solver;
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};

fn bench_nester_blf(c: &mut Criterion) {
    let mut group = c.benchmark_group("nester2d_blf");
    group.sample_size(10);

    for &n in &[5, 10, 20] {
        let geometries: Vec<Geometry2D> = (0..n)
            .map(|i| {
                let w = 20.0 + (i as f64 * 3.0) % 30.0;
                let h = 15.0 + (i as f64 * 7.0) % 25.0;
                Geometry2D::rectangle(&format!("R{}", i), w, h)
            })
            .collect();
        let boundary = Boundary2D::rectangle(200.0, 200.0);
        let nester = Nester2D::default_config();

        group.bench_with_input(
            BenchmarkId::new("rectangles", n),
            &(geometries, boundary, nester),
            |b, (g, bd, n)| {
                b.iter(|| {
                    let result = n.solve(black_box(g), black_box(bd));
                    black_box(result)
                })
            },
        );
    }
    group.finish();
}

fn bench_geometry_creation(c: &mut Criterion) {
    c.bench_function("geometry2d_rectangle", |b| {
        b.iter(|| Geometry2D::rectangle(black_box("test"), black_box(100.0), black_box(50.0)))
    });
}

criterion_group!(benches, bench_nester_blf, bench_geometry_creation);
criterion_main!(benches);
