//! Configuration for cutting path optimization.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use u_nesting_core::error::whole_at_least;
use u_nesting_core::Error;

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

    /// Weight factor for pierce count in the cost function, a finite number
    /// at least 0 (0 ignores pierces; a negative weight would reward them).
    /// Cost = total_rapid_distance + pierce_weight * pierce_count
    pub pierce_weight: f64,

    /// Maximum number of 2-opt improvement iterations.
    /// Set to 0 to use only the nearest-neighbor solution.
    pub max_2opt_iterations: usize,

    /// Wall-clock budget (in milliseconds) for the 2-opt improvement phase.
    ///
    /// The 2-opt neighborhood is `O(n^2)` candidate moves per pass, and each
    /// pass may repeat up to `max_2opt_iterations` times, so on large inputs
    /// the iteration cap alone can still run for many seconds — blocking the
    /// (browser main) thread. When this budget is exceeded the optimizer stops
    /// and returns the best sequence found so far; cut-path order is a
    /// heuristic, so early termination never invalidates the result.
    ///
    /// Set to `0` for no wall-clock limit (bounded only by
    /// `max_2opt_iterations`). Default: `5000` (5 seconds).
    pub time_limit_ms: u64,

    /// Machine rapid traverse speed (mm/s or units/s), a finite number above
    /// 0. Used only for time estimation, not for optimization.
    pub rapid_speed: f64,

    /// Machine cutting speed (mm/s or units/s), a finite number above 0.
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

    /// Tolerance for geometric comparisons, a finite number at least 0: the
    /// slack allowed when a gap between two parts is compared with the kerf
    /// width (a gap within `tolerance` of the kerf is wide enough) and when
    /// edges are matched as shared.
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

impl CutDirectionPreference {
    /// Parses `"ccw"`, `"cw"` or `"auto"` (case-insensitive); `None` for any
    /// other name, so a misspelling is refused rather than read as `Auto`.
    /// [`Self::parse`], or `unknown_option` naming `parameter` and the names
    /// it accepts.
    pub fn read(parameter: &str, name: &str) -> u_nesting_core::Result<Self> {
        Self::parse(name).ok_or_else(|| Error::UnknownOption {
            parameter: parameter.to_string(),
            got: name.to_string(),
            expected: vec!["ccw".into(), "cw".into(), "auto".into()],
        })
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "ccw" => Some(Self::Ccw),
            "cw" => Some(Self::Cw),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

impl Default for CuttingConfig {
    fn default() -> Self {
        Self {
            kerf_width: 0.0,
            pierce_weight: 10.0,
            max_2opt_iterations: 1000,
            time_limit_ms: 5000,
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

    /// Refuses a configuration the optimizer would otherwise reinterpret, as
    /// `parameter_out_of_range`: a kerf width that is negative or not finite
    /// (a negative one used to turn compensation off), a `pierce_weight` or
    /// `tolerance` that is negative or not finite, a `cut_speed` or
    /// `rapid_speed` that is not above 0 (the time estimate divides by them),
    /// and fewer than one pierce candidate. The builders store what they are
    /// given; this is where a value is judged.
    pub fn validate(&self) -> u_nesting_core::Result<()> {
        use u_nesting_core::error::{check_above, check_at_least};
        if !(self.kerf_width.is_finite() && self.kerf_width >= 0.0) {
            return Err(Error::OutOfRange {
                parameter: "kerf_width".to_string(),
                id: None,
                min: Some(0.0),
                max: None,
                exclusive_min: None,
                exclusive_max: None,
                got: self.kerf_width,
                range: "finite and >= 0 (0 disables compensation)".to_string(),
            });
        }
        check_at_least("pierce_weight", self.pierce_weight, 0.0)?;
        check_above("rapid_speed", self.rapid_speed, 0.0)?;
        check_above("cut_speed", self.cut_speed, 0.0)?;
        Self::check_pierce_candidates(self.pierce_candidates as i128)?;
        check_at_least("tolerance", self.tolerance, 0.0)?;
        Ok(())
    }

    /// A number of pierce candidates: at least 1. A request's value and a
    /// configuration's own are judged here alike.
    pub(crate) fn check_pierce_candidates(value: i128) -> u_nesting_core::Result<usize> {
        whole_at_least("pierce_candidates", value, 1)
    }

    /// The configuration a cutting request asks for, refused as the optimizer
    /// would refuse it: an unknown direction name is `unknown_option`, a value
    /// outside its range `parameter_out_of_range`.
    #[cfg(feature = "serde")]
    pub fn from_request(
        request: Option<&u_nesting_core::api_types::CuttingConfigRequest>,
    ) -> u_nesting_core::Result<Self> {
        let mut config = Self::default();
        if let Some(req) = request {
            if let Some(kerf) = req.kerf_width {
                config.kerf_width = kerf;
            }
            if let Some(weight) = req.pierce_weight {
                config.pierce_weight = weight;
            }
            if let Some(iters) = req.max_2opt_iterations {
                config.max_2opt_iterations = whole_at_least("max_2opt_iterations", iters, 0)?;
            }
            if let Some(time_limit) = req.time_limit_ms {
                config.time_limit_ms = whole_at_least("time_limit_ms", time_limit, 0)?;
            }
            if let Some(speed) = req.rapid_speed {
                config.rapid_speed = speed;
            }
            if let Some(speed) = req.cut_speed {
                config.cut_speed = speed;
            }
            if let Some(dir) = &req.exterior_direction {
                config.exterior_direction =
                    CutDirectionPreference::read("exterior_direction", dir)?;
            }
            if let Some(dir) = &req.interior_direction {
                config.interior_direction =
                    CutDirectionPreference::read("interior_direction", dir)?;
            }
            if let Some(home) = req.home_position {
                config.home_position = (home[0], home[1]);
            }
            if let Some(candidates) = req.pierce_candidates {
                config.pierce_candidates = Self::check_pierce_candidates(candidates)?;
            }
            if let Some(tol) = req.tolerance {
                config.tolerance = tol;
            }
        }
        config.validate()?;
        Ok(config)
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

    /// Sets the wall-clock budget (ms) for the 2-opt phase. `0` = unlimited.
    pub fn with_time_limit_ms(mut self, time_limit_ms: u64) -> Self {
        self.time_limit_ms = time_limit_ms;
        self
    }

    /// Sets the home position.
    pub fn with_home_position(mut self, x: f64, y: f64) -> Self {
        self.home_position = (x, y);
        self
    }

    /// Sets the number of pierce candidates per contour.
    pub fn with_pierce_candidates(mut self, candidates: usize) -> Self {
        self.pierce_candidates = candidates;
        self
    }

    /// Sets the machine speeds (units/s) the time estimate uses.
    pub fn with_speeds(mut self, rapid_speed: f64, cut_speed: f64) -> Self {
        self.rapid_speed = rapid_speed;
        self.cut_speed = cut_speed;
        self
    }

    /// Sets the tolerance for geometric comparisons.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
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
        assert_eq!(config.time_limit_ms, 5000);
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

    /// The builders store what they are given; `validate` refuses it.
    #[test]
    fn a_setting_out_of_range_is_stored_and_refused() {
        let config = CuttingConfig::new().with_pierce_candidates(0);
        assert_eq!(config.pierce_candidates, 0, "stored as given");
        let e = config.validate().expect_err("no pierce candidate");
        assert_eq!(e.code(), "parameter_out_of_range");

        let refused = |config: CuttingConfig| match config.validate() {
            Err(Error::OutOfRange {
                parameter,
                min,
                exclusive_min,
                got,
                ..
            }) => (parameter, min, exclusive_min, got),
            other => panic!("{other:?}"),
        };
        for (rapid, cut, parameter) in [
            (0.0, 100.0, "rapid_speed"),
            (-1.0, 100.0, "rapid_speed"),
            (1000.0, 0.0, "cut_speed"),
            (1000.0, -1.0, "cut_speed"),
        ] {
            let got = if parameter == "rapid_speed" {
                rapid
            } else {
                cut
            };
            assert_eq!(
                refused(CuttingConfig::new().with_speeds(rapid, cut)),
                (parameter.to_string(), None, Some(0.0), got)
            );
        }
        assert_eq!(
            refused(CuttingConfig::new().with_pierce_weight(-1.0)),
            ("pierce_weight".to_string(), Some(0.0), None, -1.0)
        );
        assert_eq!(
            refused(CuttingConfig::new().with_tolerance(-1e-6)),
            ("tolerance".to_string(), Some(0.0), None, -1e-6)
        );
        assert!(CuttingConfig::new()
            .with_tolerance(f64::NAN)
            .validate()
            .is_err());
        assert!(CuttingConfig::new()
            .with_pierce_weight(0.0)
            .with_tolerance(0.0)
            .validate()
            .is_ok());
    }
}
