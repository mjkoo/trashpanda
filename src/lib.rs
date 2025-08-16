//! TrashPanda: A Rust library for contextual multi-armed bandits.
//!
//! This library provides efficient implementations of various multi-armed bandit
//! algorithms, inspired by the Python MABWiser library but designed with Rust's
//! performance and safety guarantees in mind.
//!
//! # Quick Start
//!
//! ```
//! use trashpanda::Bandit;
//! use trashpanda::policies::EpsilonGreedy;
//!
//! // Create a bandit with three arms directly
//! let mut bandit = Bandit::new(
//!     vec!["red", "blue", "green"],
//!     EpsilonGreedy::new(0.1)
//! ).unwrap();
//!
//! // Or use convenience constructor
//! let mut bandit = Bandit::epsilon_greedy(
//!     vec!["red", "blue", "green"],
//!     0.1
//! ).unwrap();
//!
//! // Train on historical data
//! let decisions = vec!["red", "blue", "red"];
//! let rewards = vec![1.0, 0.5, 0.8];
//! // bandit.fit(&decisions, &rewards).unwrap();
//!
//! // Make a prediction
//! // let mut rng = rand::thread_rng();
//! // let choice = bandit.predict(&mut rng).unwrap();
//! ```

// #![warn(missing_docs)]
// #![warn(clippy::all)]

mod bandit;
mod error;
pub mod policies;
mod regression;

// Re-export main types
pub use bandit::Bandit;
pub use error::{BanditError, Result};

// Re-export IndexSet for users implementing custom policies
pub use indexmap::IndexSet;

/// Prelude module for convenient imports.
///
/// # Examples
///
/// ```
/// use trashpanda::prelude::*;
/// ```
pub mod prelude {
    pub use crate::policies::{
        EpsilonGreedy, LinGreedy, LinTs, LinUcb, Policy, Random, ThompsonSampling, Ucb,
    };
    pub use crate::{Bandit, BanditError, Result};
    // IndexSet is available for custom policy implementations
    pub use indexmap::IndexSet;
}
