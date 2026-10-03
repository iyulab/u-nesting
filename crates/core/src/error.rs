//! Error types for U-Nesting.

use thiserror::Error;

/// Result type alias for U-Nesting operations.
pub type Result<T> = std::result::Result<T, Error>;

/// `value` inside `[min, max]` (a NaN is outside every range), or a
/// `ConfigError` naming the field and the range.
pub(crate) fn check_range(name: &str, value: f64, min: f64, max: f64) -> Result<()> {
    if value >= min && value <= max {
        Ok(())
    } else {
        Err(Error::ConfigError(format!(
            "{name} must be in [{min}, {max}], got {value}"
        )))
    }
}

/// `value` finite and at least `min`, or a `ConfigError` naming the field.
pub(crate) fn check_at_least(name: &str, value: f64, min: f64) -> Result<()> {
    if value.is_finite() && value >= min {
        Ok(())
    } else {
        Err(Error::ConfigError(format!(
            "{name} must be a finite number >= {min}, got {value}"
        )))
    }
}

/// Errors that can occur during nesting/packing operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid geometry provided.
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),

    /// Invalid boundary provided.
    #[error("Invalid boundary: {0}")]
    InvalidBoundary(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// NFP computation failed.
    #[error("NFP computation failed: {0}")]
    NfpError(String),

    /// No valid placement found.
    #[error("No valid placement found for geometry: {0}")]
    NoPlacement(String),

    /// Computation cancelled.
    #[error("Computation cancelled")]
    Cancelled,

    /// Timeout exceeded.
    #[error("Timeout exceeded after {0}ms")]
    Timeout(u64),

    /// Serialization error.
    #[cfg(feature = "serde")]
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}
