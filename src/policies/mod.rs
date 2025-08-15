pub mod epsilon_greedy;
pub mod random;
pub mod thompson;
pub mod ucb;

pub use epsilon_greedy::EpsilonGreedy;
pub use random::Random;
pub use thompson::ThompsonSampling;
pub use ucb::Ucb;

use indexmap::IndexSet;
use std::collections::HashMap;

/// Core trait for bandit learning policies
///
/// This trait no longer requires Send + Sync bounds, as they're only
/// needed when using dynamic dispatch with threading. The generic
/// Bandit<A, P> design allows the compiler to determine these bounds
/// based on actual usage.
///
/// The trait uses `IndexSet` for the arms collection to provide O(1) lookups
/// and indexed access while preserving insertion order.
pub trait Policy<A> {
    /// Update the policy with observed rewards for decisions
    fn update(&mut self, decisions: &[A], rewards: &[f64]);

    /// Select an arm from the available arms using a random source
    fn select(&self, arms: &IndexSet<A>, rng: &mut dyn rand::RngCore) -> Option<A>;

    /// Get the expected reward probability for each arm
    fn expectations(&self, arms: &IndexSet<A>) -> HashMap<A, f64>;

    /// Reset statistics for a specific arm (e.g., when arm is removed/re-added)
    fn reset_arm(&mut self, arm: &A);

    /// Reset all statistics
    fn reset(&mut self);
}
