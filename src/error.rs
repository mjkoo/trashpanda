//! Error types for the trashpanda library.

use thiserror::Error;

/// Result type alias for bandit operations.
pub type Result<T> = std::result::Result<T, BanditError>;

/// Result type alias for policy operations.
pub type PolicyResult<T> = std::result::Result<T, PolicyError>;

/// Errors that can occur during bandit operations.
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
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

    /// Builder configuration error.
    #[error("builder error: {message}")]
    BuilderError { message: String },

    /// Policy-related error.
    #[error(transparent)]
    PolicyError(#[from] PolicyError),
}

/// Errors that can occur during policy operations.
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PolicyError {
    /// Context was provided to a non-contextual policy.
    #[error("this policy does not support context (use fit() and predict() instead)")]
    ContextNotSupported,

    /// Context was required but not provided.
    #[error("this policy requires context (use fit_with_context() and predict_with_context())")]
    ContextRequired,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BanditError::ArmNotFound;
        assert_eq!(err.to_string(), "arm not found");

        let err = BanditError::DimensionMismatch {
            message: "context dimensions mismatch".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "dimension mismatch: context dimensions mismatch"
        );

        let err = BanditError::BuilderError {
            message: "invalid configuration".to_string(),
        };
        assert_eq!(err.to_string(), "builder error: invalid configuration");
    }
}
