//! Configuration for cutting path optimization.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::bridge::BridgeConfig;
use crate::leadin::LeadInConfig;
use crate::thermal::ThermalConfig;

/// Configuration parameters for cutting path optimization.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CuttingConfig {
    /// Kerf width (cutting tool width) in the same units as geometry coordinates.
    /// Used for kerf compensation. Set to 0.0 to disable.
    pub kerf_width: f64,

    /// Weight factor for pierce count in the cost function.
    /// Cost = total_rapid_distance + pierce_weight * pierce_count
    pub pierce_weight: f64,

    /// Maximum number of 2-opt improvement iterations.
    /// Set to 0 to use only the nearest-neighbor solution.
    pub max_2opt_iterations: usize,

    /// Machine rapid traverse speed (mm/s or units/s).
    /// Used only for time estimation, not for optimization.
    pub rapid_speed: f64,

    /// Machine cutting speed (mm/s or units/s).
    /// Used only for time estimation, not for optimization.
    pub cut_speed: f64,

    /// Default cut direction for exterior contours.
    pub exterior_direction: CutDirectionPreference,

    /// Default cut direction for interior contours (holes).
    pub interior_direction: CutDirectionPreference,

    /// Home position for the cutting head (start/end point).
    /// Default is (0.0, 0.0).
    pub home_position: (f64, f64),

    /// Number of candidate pierce points to evaluate per contour.
    /// Higher values give better pierce point selection but slower optimization.
    /// Default: 1 (use nearest point on contour to previous endpoint).
    pub pierce_candidates: usize,

    /// Tolerance for geometric comparisons.
    pub tolerance: f64,

    /// Lead-in/lead-out configuration.
    pub lead_in: LeadInConfig,

    /// Bridge/tab (micro-joint) configuration.
    pub bridge: BridgeConfig,

    /// Thermal model configuration.
    pub thermal: ThermalConfig,
}

/// Preference for cutting direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CutDirectionPreference {
    /// Counter-clockwise (conventional for exterior contours).
    Ccw,
    /// Clockwise (conventional for interior/hole contours).
    Cw,
    /// Automatically determine based on contour type.
    Auto,
}

impl Default for CuttingConfig {
    fn default() -> Self {
        Self {
            kerf_width: 0.0,
            pierce_weight: 10.0,
            max_2opt_iterations: 1000,
            rapid_speed: 1000.0,
            cut_speed: 100.0,
            exterior_direction: CutDirectionPreference::Auto,
            interior_direction: CutDirectionPreference::Auto,
            home_position: (0.0, 0.0),
            pierce_candidates: 1,
            tolerance: 1e-6,
            lead_in: LeadInConfig::default(),
            bridge: BridgeConfig::default(),
            thermal: ThermalConfig::default(),
        }
    }
}

impl CuttingConfig {
    /// Creates a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the kerf width.
    pub fn with_kerf_width(mut self, width: f64) -> Self {
        self.kerf_width = width;
        self
    }

    /// Sets the pierce weight factor.
    pub fn with_pierce_weight(mut self, weight: f64) -> Self {
        self.pierce_weight = weight;
        self
    }

    /// Sets the maximum 2-opt iterations.
    pub fn with_max_2opt_iterations(mut self, iterations: usize) -> Self {
        self.max_2opt_iterations = iterations;
        self
    }

    /// Sets the home position.
    pub fn with_home_position(mut self, x: f64, y: f64) -> Self {
        self.home_position = (x, y);
        self
    }

    /// Sets the number of pierce candidates per contour.
    pub fn with_pierce_candidates(mut self, candidates: usize) -> Self {
        self.pierce_candidates = candidates.max(1);
        self
    }

    /// Sets the lead-in/lead-out configuration.
    pub fn with_lead_in(mut self, lead_in: LeadInConfig) -> Self {
        self.lead_in = lead_in;
        self
    }

    /// Sets the bridge/tab configuration.
    pub fn with_bridge(mut self, bridge: BridgeConfig) -> Self {
        self.bridge = bridge;
        self
    }

    /// Sets the thermal model configuration.
    pub fn with_thermal(mut self, thermal: ThermalConfig) -> Self {
        self.thermal = thermal;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CuttingConfig::default();
        assert_eq!(config.kerf_width, 0.0);
        assert_eq!(config.pierce_weight, 10.0);
        assert_eq!(config.max_2opt_iterations, 1000);
        assert_eq!(config.home_position, (0.0, 0.0));
    }

    #[test]
    fn test_builder() {
        let config = CuttingConfig::new()
            .with_kerf_width(0.5)
            .with_pierce_weight(20.0)
            .with_home_position(10.0, 10.0);

        assert_eq!(config.kerf_width, 0.5);
        assert_eq!(config.pierce_weight, 20.0);
        assert_eq!(config.home_position, (10.0, 10.0));
    }

    #[test]
    fn test_pierce_candidates_minimum() {
        let config = CuttingConfig::new().with_pierce_candidates(0);
        assert_eq!(config.pierce_candidates, 1);
    }
}
