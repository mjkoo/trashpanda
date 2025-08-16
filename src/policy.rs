//! Policy trait definition for multi-armed bandit algorithms

use std::collections::HashMap;

use indexmap::IndexSet;

/// Unified trait for both contextual and non-contextual bandit policies
///
/// This trait provides a unified interface for both contextual and non-contextual
/// policies through a generic context parameter. Non-contextual policies use
/// the default `()` type passed by value, while contextual policies typically
/// use reference types like `&[f64]`.
///
/// # Type Parameters
/// - `A`: The arm type, must be hashable
/// - `C`: The context type, defaults to `()` for non-contextual policies
///
/// The trait uses `IndexSet` for the arms collection to provide O(1) lookups
/// and indexed access while preserving insertion order.
pub trait Policy<A, C = ()> {
    /// Update the policy with an observed reward for a single decision
    ///
    /// # Arguments
    /// - `decision`: The arm that was selected
    /// - `context`: The context when the decision was made
    /// - `reward`: The observed reward
    fn update(&mut self, decision: &A, context: C, reward: f64);

    /// Select an arm from the available arms given a context
    ///
    /// # Arguments
    /// - `arms`: The available arms to choose from
    /// - `context`: The current context
    /// - `rng`: Random number generator for stochastic policies
    ///
    /// # Returns
    /// The selected arm, or None if no arms are available
    fn select(&self, arms: &IndexSet<A>, context: C, rng: &mut dyn rand::RngCore) -> Option<A>;

    /// Get the expected reward for each arm given a context
    ///
    /// # Arguments
    /// - `arms`: The available arms
    /// - `context`: The current context
    /// - `rng`: Random number generator for stochastic expectations (e.g., epsilon-greedy mixing)
    ///
    /// # Returns
    /// A map of arms to their expected rewards
    fn expectations(
        &self,
        arms: &IndexSet<A>,
        context: C,
        rng: &mut dyn rand::RngCore,
    ) -> HashMap<A, f64>;

    /// Reset statistics for a specific arm (e.g., when arm is removed/re-added)
    fn reset_arm(&mut self, arm: &A);

    /// Reset all statistics
    fn reset(&mut self);
}
