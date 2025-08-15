pub mod epsilon_greedy;
pub mod linucb;
pub mod random;
pub mod thompson;
pub mod ucb;

pub use epsilon_greedy::EpsilonGreedy;
pub use linucb::LinUcb;
pub use random::Random;
pub use thompson::ThompsonSampling;
pub use ucb::Ucb;

use crate::error::PolicyResult;
use indexmap::IndexSet;
use std::collections::HashMap;

/// Unified trait for both contextual and non-contextual bandit policies
///
/// This trait unifies the previous separate Policy and ContextualPolicy traits.
/// Non-contextual policies should implement the base methods and return errors
/// for the _with_context methods. Contextual policies should implement the
/// _with_context methods.
///
/// The trait uses `IndexSet` for the arms collection to provide O(1) lookups
/// and indexed access while preserving insertion order.
pub trait Policy<A> {
    /// Update the policy with observed rewards for decisions
    ///
    /// Non-contextual policies should implement this method.
    fn update(&mut self, decisions: &[A], rewards: &[f64]);

    /// Update the policy with observed rewards and contexts
    ///
    /// Contextual policies should implement this method.
    /// Non-contextual policies should return PolicyError::ContextNotSupported.
    fn update_with_context(
        &mut self,
        decisions: &[A],
        contexts: Option<&[Vec<f64>]>,
        rewards: &[f64],
    ) -> PolicyResult<()>;

    /// Select an arm from the available arms using a random source
    ///
    /// Non-contextual policies should implement this method.
    fn select(&self, arms: &IndexSet<A>, rng: &mut dyn rand::RngCore) -> Option<A>;

    /// Select an arm given a context
    ///
    /// Contextual policies should implement this method.
    /// Non-contextual policies should return PolicyError::ContextNotSupported.
    fn select_with_context(
        &self,
        arms: &IndexSet<A>,
        context: Option<&[f64]>,
        rng: &mut dyn rand::RngCore,
    ) -> PolicyResult<Option<A>>;

    /// Get the expected reward probability for each arm
    ///
    /// Non-contextual policies should implement this method.
    fn expectations(&self, arms: &IndexSet<A>) -> HashMap<A, f64>;

    /// Get the expected reward for each arm given a context
    ///
    /// Contextual policies should implement this method.
    /// Non-contextual policies should return PolicyError::ContextNotSupported.
    fn expectations_with_context(
        &self,
        arms: &IndexSet<A>,
        context: Option<&[f64]>,
    ) -> PolicyResult<HashMap<A, f64>>;

    /// Reset statistics for a specific arm (e.g., when arm is removed/re-added)
    fn reset_arm(&mut self, arm: &A);

    /// Reset all statistics
    fn reset(&mut self);

    /// Returns true if this policy requires context features
    ///
    /// Default implementation returns false (for non-contextual policies)
    fn requires_context(&self) -> bool {
        false
    }

    /// Returns the number of features expected (0 for non-contextual)
    ///
    /// Default implementation returns 0 (for non-contextual policies)
    fn num_features(&self) -> usize {
        0
    }
}
