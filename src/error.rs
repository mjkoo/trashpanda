//! Error types for the trashpanda library.

use thiserror::Error;

/// Result type alias for bandit operations.
pub type Result<T> = std::result::Result<T, BanditError>;

/// Errors that can occur during bandit operations.
#[derive(Error, Debug)]
pub enum BanditError {
    /// The specified arm was not found in the bandit.
    #[error("arm not found")]
    ArmNotFound,

    /// The specified arm already exists in the bandit.
    #[error("arm already exists")]
    ArmAlreadyExists,

    /// No arms are available in the bandit.
    #[error("no arms available")]
    NoArmsAvailable,

    /// Mismatch in the dimensions of input data.
    #[error("dimension mismatch: {message}")]
    DimensionMismatch { message: String },

    /// Invalid parameter value.
    #[error("invalid parameter: {message}")]
    InvalidParameter { message: String },

    /// The bandit has not been trained yet.
    #[error("bandit not trained: no data has been fitted")]
    NotTrained,

    /// Invalid context dimensions.
    #[error("invalid context dimensions: expected {expected}, got {got}")]
    InvalidContextDimensions { expected: usize, got: usize },

    /// Numerical computation error.
    #[error("numerical error: {message}")]
    NumericalError { message: String },

    /// Builder configuration error.
    #[error("builder error: {message}")]
    BuilderError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BanditError::ArmNotFound;
        assert_eq!(err.to_string(), "arm not found");

        let err = BanditError::InvalidParameter {
            message: "epsilon must be between 0 and 1".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "invalid parameter: epsilon must be between 0 and 1"
        );

        let err = BanditError::InvalidContextDimensions {
            expected: 10,
            got: 5,
        };
        assert_eq!(
            err.to_string(),
            "invalid context dimensions: expected 10, got 5"
        );
    }
}
