//! Low-level polygon predicates shared by boundary containment (out-of-bounds
//! filtering) and input validation (self-intersection rejection).
//!
//! `u-geometry` supplies area / centroid / point-in-polygon primitives but no
//! public segment-segment intersection predicate, which both the polygon
//! boundary-containment check and the simple-polygon check need. These helpers
//! provide it with an orientation (signed-area) test.

/// Absolute threshold below which an orientation determinant is treated as zero
/// (collinear). Cross products here are in coordinate-squared units; nesting
/// coordinates are typically 1e0–1e4, so this tolerates float noise without
/// masking genuine crossings.
const ORIENT_EPS: f64 = 1e-9;

/// Sign of the cross product `(b - a) × (c - a)`:
/// `+1` if `a→b→c` turns counter-clockwise, `-1` if clockwise, `0` if collinear.
fn orientation(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> i32 {
    let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    if cross > ORIENT_EPS {
        1
    } else if cross < -ORIENT_EPS {
        -1
    } else {
        0
    }
}

/// Returns true when point `p`, known to be collinear with segment `a→b`, lies
/// within that segment's bounding box (i.e. actually on the segment).
fn on_segment(a: (f64, f64), b: (f64, f64), p: (f64, f64)) -> bool {
    p.0 >= a.0.min(b.0) - ORIENT_EPS
        && p.0 <= a.0.max(b.0) + ORIENT_EPS
        && p.1 >= a.1.min(b.1) - ORIENT_EPS
        && p.1 <= a.1.max(b.1) + ORIENT_EPS
}

/// Returns true when closed segments `p1→p2` and `p3→p4` intersect, including
/// endpoint-touching and collinear overlap (the general/degenerate cases).
pub(crate) fn segments_intersect(
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    p4: (f64, f64),
) -> bool {
    let d1 = orientation(p3, p4, p1);
    let d2 = orientation(p3, p4, p2);
    let d3 = orientation(p1, p2, p3);
    let d4 = orientation(p1, p2, p4);

    // Proper crossing: each segment straddles the other's supporting line.
    if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) {
        return true;
    }

    // Collinear / touching endpoints.
    (d1 == 0 && on_segment(p3, p4, p1))
        || (d2 == 0 && on_segment(p3, p4, p2))
        || (d3 == 0 && on_segment(p1, p2, p3))
        || (d4 == 0 && on_segment(p1, p2, p4))
}

/// Reflects a polygon across the y-axis (`(x, y) -> (-x, y)`) and reverses
/// vertex order to restore CCW winding — a reflection has determinant -1, so
/// leaving the order unchanged would silently flip an originally-CCW ring to
/// CW, breaking every downstream signed-area/orientation assumption.
///
/// Used for `allow_flip` mirroring: standard technique (Bennell & Oliveira
/// 2008) is to reflect the orbiting polygon once and feed it through the
/// existing NFP/placement pipeline as an additional orientation candidate,
/// the same way rotation candidates are enumerated.
pub(crate) fn mirror_polygon(vertices: &[(f64, f64)]) -> Vec<(f64, f64)> {
    vertices.iter().rev().map(|&(x, y)| (-x, y)).collect()
}

/// Returns true when the polygon ring is **simple**: no pair of non-adjacent
/// edges intersects. Adjacent edges (sharing a vertex) are exempt, as is the
/// closing wrap-around pair. Fewer than 3 vertices is vacuously not simple.
///
/// O(n²) in the vertex count — run once at input validation, not per placement.
pub(crate) fn is_simple_polygon(vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }

    for i in 0..n {
        let a1 = vertices[i];
        let a2 = vertices[(i + 1) % n];
        for j in (i + 1)..n {
            // Adjacent edges share a vertex and are exempt: the immediate
            // successor (j == i+1) and the closing wrap-around pair (i==0, j==n-1).
            if j == i + 1 || (i == 0 && j == n - 1) {
                continue;
            }
            let b1 = vertices[j];
            let b2 = vertices[(j + 1) % n];
            if segments_intersect(a1, a2, b1, b2) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_segments_intersect() {
        assert!(segments_intersect(
            (0.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (10.0, 0.0)
        ));
    }

    #[test]
    fn disjoint_segments_do_not_intersect() {
        assert!(!segments_intersect(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 5.0),
            (1.0, 5.0)
        ));
    }

    #[test]
    fn square_is_simple() {
        let sq = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(is_simple_polygon(&sq));
    }

    #[test]
    fn bowtie_is_not_simple() {
        // Self-intersecting "bowtie": edges (0,0)-(100,100) and (100,0)-(0,100) cross.
        let bowtie = [(0.0, 0.0), (100.0, 100.0), (100.0, 0.0), (0.0, 100.0)];
        assert!(!is_simple_polygon(&bowtie));
    }

    #[test]
    fn mirror_polygon_preserves_ccw_winding() {
        // CCW unit square.
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert_eq!(
            orientation(square[0], square[1], square[2]),
            1,
            "fixture must be CCW"
        );

        let mirrored = mirror_polygon(&square);
        assert_eq!(
            orientation(mirrored[0], mirrored[1], mirrored[2]),
            1,
            "mirroring must restore CCW winding, not leave it CW"
        );
    }

    #[test]
    fn mirror_polygon_reflects_across_y_axis() {
        // Asymmetric right triangle, order-independent point-set comparison
        // (mirroring reverses traversal order alongside reflecting x).
        let tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 2.0)];
        let mirrored = mirror_polygon(&tri);
        let mut expected: Vec<(f64, f64)> = tri.iter().map(|&(x, y)| (-x, y)).collect();
        let mut got = mirrored.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, expected);
    }

    #[test]
    fn mirror_polygon_is_involution() {
        // Mirroring twice must return to the original polygon (up to the
        // winding-restoring reversal, which is itself an involution on order).
        let l = [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (10.0, 10.0),
            (10.0, 20.0),
            (0.0, 20.0),
        ];
        let twice = mirror_polygon(&mirror_polygon(&l));
        assert_eq!(twice, l);
    }

    #[test]
    fn concave_l_shape_is_simple() {
        let l = [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (10.0, 10.0),
            (10.0, 20.0),
            (0.0, 20.0),
        ];
        assert!(is_simple_polygon(&l));
    }
}
