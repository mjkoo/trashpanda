//! TrashPanda: A Rust library for contextual multi-armed bandits.
//!
//! This library provides efficient implementations of various multi-armed bandit
//! algorithms, inspired by the Python MABWiser library but designed with Rust's
//! performance and safety guarantees in mind.
//!
//! # Quick Start
//!
//! ```
//! use trashpanda::{Bandit, LearningPolicy, Arm};
//!
//! // Create a bandit with three arms
//! let mut bandit = Bandit::builder()
//!     .arms(vec!["red", "blue", "green"])
//!     .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
//!     .build()
//!     .unwrap();
//!
//! // Train on historical data
//! let decisions = vec![Arm::from("red"), Arm::from("blue"), Arm::from("red")];
//! let rewards = vec![1.0, 0.5, 0.8];
//! // bandit.fit(&decisions, &rewards).unwrap();
//!
//! // Make a prediction
//! // let choice = bandit.predict().unwrap();
//! ```

// #![warn(missing_docs)]
// #![warn(clippy::all)]

mod arm;
mod bandit;
mod error;

// Re-export main types
pub use arm::Arm;
pub use bandit::{ArmMetadata, Bandit, BanditBuilder, LearningPolicy};
pub use error::{BanditError, Result};

/// Prelude module for convenient imports.
///
/// # Examples
///
/// ```
/// use trashpanda::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Arm, Bandit, BanditError, LearningPolicy, Result};
}
