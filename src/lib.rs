//! TrashPanda: A Rust library for contextual multi-armed bandits.
//!
//! This library provides efficient implementations of various multi-armed bandit
//! algorithms, inspired by the Python MABWiser library but designed with Rust's
//! performance and safety guarantees in mind.
//!
//! # Quick Start
//!
//! ```
//! use trashpanda::{Bandit, LearningPolicy};
//!
//! // Create a bandit with three arms
//! let mut bandit = Bandit::builder()
//!     .arms(vec!["red", "blue", "green"])
//!     .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
//!     .build()
//!     .unwrap();
//!
//! // Train on historical data
//! let decisions = vec!["red", "blue", "red"];
//! let rewards = vec![1.0, 0.5, 0.8];
//! // bandit.fit(&decisions, &rewards).unwrap();
//!
//! // Make a prediction
//! // let choice = bandit.predict().unwrap();
//! ```

// #![warn(missing_docs)]
// #![warn(clippy::all)]

mod bandit;
mod error;
mod policies;

// Re-export main types
pub use bandit::{ArmMetadata, Bandit, BanditBuilder, LearningPolicy};
pub use error::{BanditError, Result};
pub use policies::Policy;

/// Prelude module for convenient imports.
///
/// # Examples
///
/// ```
/// use trashpanda::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Bandit, BanditError, LearningPolicy, Result};
}
