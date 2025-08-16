//! TrashPanda: A Rust library for contextual multi-armed bandits.
//!
//! This library provides efficient implementations of various multi-armed bandit
//! algorithms, inspired by the Python MABWiser library but designed with Rust's
//! performance and safety guarantees in mind.
//!
//! # Quick Start
//!
//! ```
//! use trashpanda::{Bandit, simple::epsilon_greedy::EpsilonGreedy};
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
//! // Access arms using convenient methods
//! println!("Number of arms: {}", bandit.arms_len());
//!
//! // Iterate over arms without exposing IndexSet
//! for arm in bandit.arms_iter() {
//!     println!("Available arm: {}", arm);
//! }
//!
//! // Get arm by index (O(1) operation)
//! if let Some(first_arm) = bandit.get_arm_by_index(0) {
//!     println!("First arm: {}", first_arm);
//! }
//!
//! // Train on historical data
//! let decisions = vec!["red", "blue", "red"];
//! let rewards = vec![1.0, 0.5, 0.8];
//! // bandit.fit_simple(&decisions, &rewards).unwrap();
//!
//! // Make a prediction
//! // let mut rng = rand::thread_rng();
//! // let choice = bandit.predict_simple(&mut rng).unwrap();
//! ```

// #![warn(missing_docs)]
// #![warn(clippy::all)]

mod bandit;
pub mod contextual;
mod error;
pub mod neighborhood;
pub mod policy;
mod regression;
pub mod simple;

// Re-export main types
pub use bandit::Bandit;
pub use error::{BanditError, Result};

// Re-export IndexSet for users implementing custom policies
pub use indexmap::IndexSet;
