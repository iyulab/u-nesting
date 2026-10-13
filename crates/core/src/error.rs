//! Error types for U-Nesting.

use thiserror::Error;

/// Result type alias for U-Nesting operations.
pub type Result<T> = std::result::Result<T, Error>;

/// `value` inside `[min, max]` (a NaN is outside every range), or an
/// [`Error::OutOfRange`] naming the field and the range.
///
/// # Errors
///
/// [`Error::OutOfRange`] with the closed bounds `min` and `max`.
pub fn check_range(name: &str, value: f64, min: f64, max: f64) -> Result<()> {
    if value >= min && value <= max {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            parameter: name.to_string(),
            id: None,
            min: Some(min),
            max: Some(max),
            exclusive_min: None,
            exclusive_max: None,
            got: value,
            range: format!("in [{min}, {max}]"),
        })
    }
}

/// `value` finite and at least `min`, or an [`Error::OutOfRange`] naming the
/// field.
///
/// # Errors
///
/// [`Error::OutOfRange`] with the closed lower bound `min`.
pub fn check_at_least(name: &str, value: f64, min: f64) -> Result<()> {
    if value.is_finite() && value >= min {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            parameter: name.to_string(),
            id: None,
            min: Some(min),
            max: None,
            exclusive_min: None,
            exclusive_max: None,
            got: value,
            range: format!("a finite number >= {min}"),
        })
    }
}

/// `value` finite and strictly above `bound`, or an [`Error::OutOfRange`]
/// naming the field with `bound` as its open lower bound (`exclusive_min`).
///
/// # Errors
///
/// [`Error::OutOfRange`] with the open lower bound `bound`.
pub fn check_above(name: &str, value: f64, bound: f64) -> Result<()> {
    if value.is_finite() && value > bound {
        Ok(())
    } else {
        Err(Error::OutOfRange {
            parameter: name.to_string(),
            id: None,
            min: None,
            max: None,
            exclusive_min: Some(bound),
            exclusive_max: None,
            got: value,
            range: format!("a finite number > {bound}"),
        })
    }
}

/// An unsigned integer type a request's whole number is read into: a count,
/// a time budget, a seed.
pub trait WholeNumber: TryFrom<i128> + Copy {
    /// The largest value the type holds.
    const MAX: i128;
}

impl WholeNumber for u32 {
    const MAX: i128 = u32::MAX as i128;
}

impl WholeNumber for u64 {
    const MAX: i128 = u64::MAX as i128;
}

impl WholeNumber for usize {
    const MAX: i128 = usize::MAX as i128;
}

/// `value` -- a whole number as a request gives it, negatives included -- as
/// `T`, when it is at least `min` and `T` holds it; otherwise an
/// [`Error::OutOfRange`] naming `parameter`.
///
/// A negative and a too-small positive value are refused alike, with the
/// closed lower bound `min`; a value above what `T` holds also carries `max`.
///
/// # Errors
///
/// [`Error::OutOfRange`] when `value < min` or `value > T::MAX`.
pub fn whole_at_least<T: WholeNumber>(parameter: &str, value: i128, min: i128) -> Result<T> {
    match T::try_from(value) {
        Ok(v) if value >= min => Ok(v),
        _ => {
            let above = value > T::MAX;
            Err(Error::OutOfRange {
                parameter: parameter.to_string(),
                id: None,
                min: Some(min as f64),
                max: above.then_some(T::MAX as f64),
                exclusive_min: None,
                exclusive_max: None,
                got: value as f64,
                range: if above {
                    format!("a whole number in [{min}, {}]", T::MAX)
                } else {
                    format!("a whole number >= {min}")
                },
            })
        }
    }
}

/// Why the engine refused a request or could not finish it. Each variant
/// carries what a caller needs to point at the input to change -- the setting,
/// the geometry id, the positions -- as fields, not only in the message;
/// [`Error::code`] is the stable name of the reason.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A setting outside the values it accepts, and `range` says it in words.
    /// `min` / `max` are the closed bounds (the value may equal them),
    /// `exclusive_min` / `exclusive_max` the open ones (it must lie strictly
    /// beyond them). A side has at most one of the two; `None` on both means
    /// that side is unbounded. `id` names the geometry the setting belongs to,
    /// when it belongs to one (a quantity).
    #[error("{parameter} must be {range}, got {got}")]
    OutOfRange {
        parameter: String,
        id: Option<String>,
        min: Option<f64>,
        max: Option<f64>,
        exclusive_min: Option<f64>,
        exclusive_max: Option<f64>,
        got: f64,
        range: String,
    },

    /// A list the request must not leave empty -- `parameter` names it
    /// (`geometries`): there is nothing to solve.
    #[error("{parameter} is empty; give at least one")]
    EmptyInput { parameter: String },

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
    /// `empty_input`, `invalid_option`, `unknown_option`, `malformed_input`,
    /// `invalid_geometry`, `duplicate_id`, `invalid_boundary`, `cancelled`
    /// or `internal`.
    pub fn code(&self) -> &'static str {
        match self {
            Error::OutOfRange { .. } => "parameter_out_of_range",
            Error::EmptyInput { .. } => "empty_input",
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

    /// The setting or request part the refusal names, when it names one.
    pub fn parameter(&self) -> Option<&str> {
        match self {
            Error::OutOfRange { parameter, .. }
            | Error::EmptyInput { parameter }
            | Error::InvalidOption { parameter, .. }
            | Error::UnknownOption { parameter, .. } => Some(parameter),
            Error::MalformedInput { parameter, .. } | Error::InvalidBoundary { parameter, .. } => {
                parameter.as_deref()
            }
            Error::DuplicateId { .. } => Some("geometries"),
            _ => None,
        }
    }

    /// This refusal with its `parameter` named by its path under `section`
    /// when it is one of `fields` -- `spacing`, a field of the request's
    /// `config`, becomes `config.spacing` -- and unchanged otherwise.
    ///
    /// The Rust API names a setting by its field, as the configuration type
    /// the caller built has it; a request names it by where it sits in the
    /// request. The wire layer, which knows the request, places it.
    pub fn located(mut self, section: &str, fields: &[&str]) -> Self {
        let parameter = match &mut self {
            Error::OutOfRange { parameter, .. }
            | Error::EmptyInput { parameter }
            | Error::InvalidOption { parameter, .. }
            | Error::UnknownOption { parameter, .. } => Some(parameter),
            Error::MalformedInput { parameter, .. } | Error::InvalidBoundary { parameter, .. } => {
                parameter.as_mut()
            }
            _ => None,
        };
        if let Some(parameter) = parameter {
            if fields.contains(&parameter.as_str()) {
                *parameter = format!("{section}.{parameter}");
            }
        }
        self
    }

    /// A list the request must not leave empty.
    pub fn empty_input(parameter: &str) -> Self {
        Error::EmptyInput {
            parameter: parameter.to_string(),
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
