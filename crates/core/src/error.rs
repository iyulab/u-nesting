//! Error types for U-Nesting.

use thiserror::Error;

/// Result type alias for U-Nesting operations.
pub type Result<T> = std::result::Result<T, Error>;

/// `value` inside `[min, max]` (a NaN is outside every range), or an
/// [`Error::OutOfRange`] naming the field and the range.
pub(crate) fn check_range(name: &str, value: f64, min: f64, max: f64) -> Result<()> {
    if value >= min && value <= max {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            parameter: name.to_string(),
            min: Some(min),
            max: Some(max),
            got: value,
            range: format!("in [{min}, {max}]"),
        })
    }
}

/// `value` finite and at least `min`, or an [`Error::OutOfRange`] naming the
/// field.
pub(crate) fn check_at_least(name: &str, value: f64, min: f64) -> Result<()> {
    if value.is_finite() && value >= min {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            parameter: name.to_string(),
            min: Some(min),
            max: None,
            got: value,
            range: format!("a finite number >= {min}"),
        })
    }
}

/// Why the engine refused a request or could not finish it. Each variant
/// carries what a caller needs to point at the input to change -- the setting,
/// the geometry id, the positions -- as fields, not only in the message;
/// [`Error::code`] is the stable name of the reason.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A setting outside the values it accepts: `min` / `max` are the bounds
    /// (`None` where unbounded) and `range` says it in words.
    #[error("{parameter} must be {range}, got {got}")]
    OutOfRange {
        parameter: String,
        min: Option<f64>,
        max: Option<f64>,
        got: f64,
        range: String,
    },

    /// A setting that cannot be used as given (a strategy this build or this
    /// dimension does not offer, ...).
    #[error("{parameter}: {message}")]
    InvalidOption { parameter: String, message: String },

    /// An option name the engine does not know: `got` is the name given and
    /// `expected` the names it accepts.
    #[error("unknown {parameter} '{got}'; expected one of {}", expected.join(", "))]
    UnknownOption {
        parameter: String,
        got: String,
        expected: Vec<String>,
    },

    /// A request of the wrong shape: not JSON, a missing or unknown key, a
    /// value of the wrong type. `parameter` names the part when it is known.
    #[error("{message}")]
    MalformedInput {
        parameter: Option<String>,
        message: String,
    },

    /// The caller cancelled the run.
    #[error("cancelled by the caller")]
    Cancelled,

    /// A geometry the engine cannot work with, named by its id when the check
    /// that refused it knows the geometry.
    #[error("Invalid geometry: {message}")]
    InvalidGeometry { id: Option<String>, message: String },

    /// Two geometries share an id, at positions `first` and `second` of the
    /// request (counting from 0).
    #[error(
        "the id '{id}' is given twice, at positions {first} and {second} of geometries \
         (counting from 0); placements name geometries by id, so every geometry needs \
         its own"
    )]
    DuplicateId {
        id: String,
        first: usize,
        second: usize,
    },

    /// A boundary the engine cannot place into; `parameter` names the field
    /// when one field is at fault.
    #[error("Invalid boundary: {message}")]
    InvalidBoundary {
        parameter: Option<String>,
        message: String,
    },

    /// A failure inside the engine, not caused by the request.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl Error {
    /// The stable name of the reason: `parameter_out_of_range`,
    /// `invalid_option`, `unknown_option`, `malformed_input`,
    /// `invalid_geometry`, `duplicate_id`, `invalid_boundary`, `cancelled`
    /// or `internal`.
    pub fn code(&self) -> &'static str {
        match self {
            Error::OutOfRange { .. } => "parameter_out_of_range",
            Error::InvalidOption { .. } => "invalid_option",
            Error::UnknownOption { .. } => "unknown_option",
            Error::MalformedInput { .. } => "malformed_input",
            Error::Cancelled => "cancelled",
            Error::InvalidGeometry { .. } => "invalid_geometry",
            Error::DuplicateId { .. } => "duplicate_id",
            Error::InvalidBoundary { .. } => "invalid_boundary",
            Error::Internal(_) => "internal",
        }
    }

    /// A setting that cannot be used as given.
    pub fn invalid_option(parameter: &str, message: String) -> Self {
        Error::InvalidOption {
            parameter: parameter.to_string(),
            message,
        }
    }

    /// A geometry refused by a check that may or may not know which one.
    pub fn invalid_geometry(id: Option<&str>, message: String) -> Self {
        Error::InvalidGeometry {
            id: id.map(str::to_string),
            message,
        }
    }

    /// A request of the wrong shape, naming the part when it is known.
    pub fn malformed(parameter: Option<&str>, message: String) -> Self {
        Error::MalformedInput {
            parameter: parameter.map(str::to_string),
            message,
        }
    }

    /// A boundary refused, naming the field when one is at fault.
    pub fn invalid_boundary(parameter: Option<&str>, message: String) -> Self {
        Error::InvalidBoundary {
            parameter: parameter.map(str::to_string),
            message,
        }
    }
}
