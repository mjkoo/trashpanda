mod epsilon_greedy;
mod random;

use std::collections::HashMap;

pub use epsilon_greedy::EpsilonGreedy;
pub use random::Random;

/// Core trait for bandit learning policies
///
/// Note: This trait uses `dyn rand::RngCore` instead of a generic parameter
/// to maintain object-safety, allowing `Box<dyn Policy<A>>` to be used.
/// The slight performance cost of dynamic dispatch is acceptable for the
/// flexibility it provides in the Bandit implementation.
pub trait Policy<A>: Send + Sync {
    /// Update the policy with observed rewards for decisions
    fn update(&mut self, decisions: &[A], rewards: &[f64]);

    /// Select an arm from the available arms using a random source
    fn select(&self, arms: &[A], rng: &mut dyn rand::RngCore) -> Option<A>;

    /// Get the expected reward probability for each arm
    fn expectations(&self, arms: &[A]) -> HashMap<A, f64>;

    /// Reset statistics for a specific arm (e.g., when arm is removed/re-added)
    fn reset_arm(&mut self, arm: &A);

    /// Reset all statistics
    fn reset(&mut self);
}
