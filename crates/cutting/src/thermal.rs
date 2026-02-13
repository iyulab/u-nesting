//! Thermal model for heat-affected zone (HAZ) management.
//!
//! Models heat accumulation during cutting sequences to penalize orders
//! that cause excessive thermal stress. When consecutive cuts are too close
//! in space and time, the material has insufficient time to cool, leading
//! to quality degradation (warping, embrittlement, discoloration).
//!
//! # Model
//!
//! Each pierce point deposits heat that decays exponentially with distance
//! and time. The accumulated heat at any point is the sum of contributions
//! from all prior cuts.
//!
//! ```text
//! Heat(p, i) = Σ_{j < i} exp(-d²(p, pⱼ) / (2 × σ²)) × exp(-Δt_j / τ)
//! ```
//!
//! where `σ` is the HAZ radius and `τ` is the cooling time constant.
//!
//! # References
//!
//! - Kim et al. (2019), "Laser cutting path optimization with minimum heat accumulation"

use crate::cost::point_distance_sq;

/// Configuration for the thermal model.
#[derive(Debug, Clone, Copy)]
pub struct ThermalConfig {
    /// Heat-affected zone radius in mm.
    /// Heat decays to 1/e at this distance from the cut.
    pub haz_radius: f64,
    /// Cooling time constant in seconds.
    /// Heat decays to 1/e after this much time.
    pub cooling_time_constant: f64,
    /// Maximum acceptable accumulated heat (arbitrary units).
    /// Exceeding this triggers a penalty.
    pub critical_heat: f64,
    /// Penalty weight for heat violations in the cost function.
    pub penalty_weight: f64,
    /// Whether the thermal model is enabled.
    pub enabled: bool,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            haz_radius: 3.0,
            cooling_time_constant: 15.0,
            critical_heat: 1.0,
            penalty_weight: 50.0,
            enabled: false,
        }
    }
}

/// Heat contribution from a single cut at a given point and time.
///
/// Uses Gaussian spatial decay × exponential temporal decay:
/// `exp(-d² / (2σ²)) × exp(-Δt / τ)`
#[inline]
pub fn heat_contribution(
    dist_sq: f64,
    elapsed_time: f64,
    haz_radius: f64,
    cooling_time: f64,
) -> f64 {
    let sigma_sq = haz_radius * haz_radius;
    let spatial = (-dist_sq / (2.0 * sigma_sq)).exp();
    let temporal = (-elapsed_time / cooling_time).exp();
    spatial * temporal
}

/// Computes the total thermal penalty for a cutting sequence.
///
/// For each pierce point in the sequence, computes the accumulated heat
/// from all prior cuts and sums the penalty for any that exceed the
/// critical threshold.
///
/// # Arguments
///
/// * `pierce_points` - Pierce points in cutting order
/// * `cut_times` - Cumulative time at each pierce (seconds from start)
/// * `config` - Thermal configuration
///
/// # Returns
///
/// Total thermal penalty (0.0 if no heat violations).
pub fn thermal_penalty(
    pierce_points: &[(f64, f64)],
    cut_times: &[f64],
    config: &ThermalConfig,
) -> f64 {
    if !config.enabled || pierce_points.len() < 2 {
        return 0.0;
    }

    let n = pierce_points.len();
    let mut total_penalty = 0.0;

    for i in 1..n {
        let mut accumulated_heat = 0.0;

        for j in 0..i {
            let dist_sq = point_distance_sq(pierce_points[i], pierce_points[j]);
            let elapsed = cut_times[i] - cut_times[j];

            accumulated_heat +=
                heat_contribution(dist_sq, elapsed, config.haz_radius, config.cooling_time_constant);
        }

        if accumulated_heat > config.critical_heat {
            let excess = accumulated_heat - config.critical_heat;
            total_penalty += config.penalty_weight * excess * excess;
        }
    }

    total_penalty
}

/// Estimates cut times for a sequence based on cutting speed and rapid speed.
///
/// For each contour in the sequence, the time includes:
/// - Rapid travel from previous end point to current pierce point
/// - Cutting the contour perimeter
///
/// # Arguments
///
/// * `pierce_points` - Pierce points in cutting order
/// * `perimeters` - Perimeter of each contour (in order)
/// * `rapid_speed` - Rapid traverse speed (mm/s)
/// * `cut_speed` - Cutting speed (mm/s)
/// * `home` - Home/start position
pub fn estimate_cut_times(
    pierce_points: &[(f64, f64)],
    perimeters: &[f64],
    rapid_speed: f64,
    cut_speed: f64,
    home: (f64, f64),
) -> Vec<f64> {
    if pierce_points.is_empty() {
        return Vec::new();
    }

    let n = pierce_points.len();
    let mut times = Vec::with_capacity(n);
    let mut current_time = 0.0;

    // First: rapid from home to first pierce
    let rapid_dist = point_distance_sq(home, pierce_points[0]).sqrt();
    current_time += rapid_dist / rapid_speed;
    times.push(current_time);

    for i in 1..n {
        // Cut previous contour
        if i - 1 < perimeters.len() {
            current_time += perimeters[i - 1] / cut_speed;
        }
        // Rapid to next pierce
        let rapid_dist = point_distance_sq(pierce_points[i - 1], pierce_points[i]).sqrt();
        current_time += rapid_dist / rapid_speed;
        times.push(current_time);
    }

    times
}

/// Computes the thermal penalty for a cutting sequence given contour data.
///
/// This is a convenience function that combines time estimation and penalty
/// computation.
pub fn sequence_thermal_penalty(
    pierce_points: &[(f64, f64)],
    perimeters: &[f64],
    rapid_speed: f64,
    cut_speed: f64,
    home: (f64, f64),
    config: &ThermalConfig,
) -> f64 {
    if !config.enabled {
        return 0.0;
    }

    let times = estimate_cut_times(pierce_points, perimeters, rapid_speed, cut_speed, home);
    thermal_penalty(pierce_points, &times, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_contribution_at_zero_distance() {
        // At the same point, right after cutting: heat = 1.0
        let heat = heat_contribution(0.0, 0.0, 3.0, 15.0);
        assert!((heat - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_heat_contribution_decays_with_distance() {
        let near = heat_contribution(1.0, 0.0, 3.0, 15.0);
        let far = heat_contribution(100.0, 0.0, 3.0, 15.0);
        assert!(near > far, "Heat should decay with distance");
        assert!(far < 0.01, "Far heat should be negligible");
    }

    #[test]
    fn test_heat_contribution_decays_with_time() {
        let fresh = heat_contribution(0.0, 0.0, 3.0, 15.0);
        let aged = heat_contribution(0.0, 60.0, 3.0, 15.0);
        assert!(fresh > aged, "Heat should decay with time");
        assert!(aged < 0.1, "Old heat should be small");
    }

    #[test]
    fn test_disabled_returns_zero() {
        let config = ThermalConfig::default(); // enabled = false
        let penalty = thermal_penalty(
            &[(0.0, 0.0), (1.0, 0.0)],
            &[0.0, 0.1],
            &config,
        );
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_distant_cuts_no_penalty() {
        let config = ThermalConfig {
            enabled: true,
            haz_radius: 3.0,
            critical_heat: 0.5,
            ..ThermalConfig::default()
        };

        // Two cuts far apart — no heat accumulation
        let points = vec![(0.0, 0.0), (100.0, 100.0)];
        let times = vec![0.0, 10.0];

        let penalty = thermal_penalty(&points, &times, &config);
        assert!(penalty < 1e-10, "Distant cuts should have no thermal penalty");
    }

    #[test]
    fn test_close_cuts_penalty() {
        let config = ThermalConfig {
            enabled: true,
            haz_radius: 3.0,
            cooling_time_constant: 15.0,
            critical_heat: 0.3,
            penalty_weight: 10.0,
        };

        // Two cuts very close together and fast
        let points = vec![(0.0, 0.0), (1.0, 0.0)];
        let times = vec![0.0, 0.1]; // Very little cooling time

        let penalty = thermal_penalty(&points, &times, &config);
        assert!(penalty > 0.0, "Close rapid cuts should incur penalty");
    }

    #[test]
    fn test_estimate_cut_times() {
        let pierce = vec![(10.0, 0.0), (20.0, 0.0)];
        let perimeters = vec![40.0, 40.0];
        let times = estimate_cut_times(&pierce, &perimeters, 1000.0, 100.0, (0.0, 0.0));

        assert_eq!(times.len(), 2);
        // First: rapid from (0,0) to (10,0) = 10mm at 1000mm/s = 0.01s
        assert!((times[0] - 0.01).abs() < 1e-6);
        // Second: cut first contour (40mm at 100mm/s = 0.4s) + rapid (10mm at 1000mm/s = 0.01s)
        assert!((times[1] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_thermal_penalty_disabled() {
        let config = ThermalConfig::default();
        let penalty = sequence_thermal_penalty(
            &[(0.0, 0.0), (1.0, 0.0)],
            &[40.0, 40.0],
            1000.0,
            100.0,
            (0.0, 0.0),
            &config,
        );
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_more_cooling_less_penalty() {
        let config = ThermalConfig {
            enabled: true,
            haz_radius: 5.0,
            cooling_time_constant: 10.0,
            critical_heat: 0.1,
            penalty_weight: 10.0,
        };

        // Fast cuts (little cooling)
        let fast_penalty = thermal_penalty(
            &[(0.0, 0.0), (2.0, 0.0)],
            &[0.0, 0.01],
            &config,
        );

        // Slow cuts (lots of cooling)
        let slow_penalty = thermal_penalty(
            &[(0.0, 0.0), (2.0, 0.0)],
            &[0.0, 60.0],
            &config,
        );

        assert!(fast_penalty > slow_penalty, "Fast cuts should have more penalty than slow");
    }

    #[test]
    fn test_default_config() {
        let config = ThermalConfig::default();
        assert!(!config.enabled);
        assert!((config.haz_radius - 3.0).abs() < 1e-10);
        assert!((config.cooling_time_constant - 15.0).abs() < 1e-10);
        assert!((config.critical_heat - 1.0).abs() < 1e-10);
        assert!((config.penalty_weight - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_sequence() {
        let config = ThermalConfig { enabled: true, ..ThermalConfig::default() };
        let penalty = thermal_penalty(&[], &[], &config);
        assert_eq!(penalty, 0.0);

        let times = estimate_cut_times(&[], &[], 1000.0, 100.0, (0.0, 0.0));
        assert!(times.is_empty());
    }
}
