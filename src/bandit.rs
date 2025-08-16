//! Unified bandit implementation that handles both contextual and context-free cases.

use crate::error::BanditError;
use crate::policies::Policy;
use indexmap::IndexSet;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// Bandit that handles both contextual and context-free scenarios
pub struct Bandit<A, P, C = ()> {
    /// Available arms
    arms: IndexSet<A>,
    /// The policy (now unified)
    policy: P,
    /// Phantom data for context type
    _phantom: PhantomData<fn() -> C>,
}

impl<A, P, C> Bandit<A, P, C>
where
    A: Clone + Eq + Hash,
    P: Policy<A, C>,
{
    /// Create a new bandit
    pub fn new<I>(arms: I, policy: P) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        let arms: IndexSet<A> = arms.into_iter().collect();

        if arms.is_empty() {
            return Err(BanditError::NoArmsAvailable);
        }

        Ok(Self {
            arms,
            policy,
            _phantom: PhantomData,
        })
    }

    /// Fit the bandit with training data (batch update)
    pub fn fit(
        &mut self,
        decisions: &[A],
        context: &C,
        rewards: &[f64],
    ) -> Result<(), BanditError> {
        if decisions.len() != rewards.len() {
            return Err(BanditError::DimensionMismatch {
                message: format!(
                    "Mismatched dimensions: decisions={}, rewards={}",
                    decisions.len(),
                    rewards.len()
                ),
            });
        }

        self.validate_decisions(decisions)?;

        // Call update() for each decision-reward pair
        for (decision, reward) in decisions.iter().zip(rewards) {
            self.policy.update(decision, context, *reward);
        }
        Ok(())
    }

    /// Incrementally fit the bandit (alias for fit)
    pub fn partial_fit(
        &mut self,
        decisions: &[A],
        context: &C,
        rewards: &[f64],
    ) -> Result<(), BanditError> {
        self.fit(decisions, context, rewards)
    }

    /// Make a prediction
    pub fn predict(&self, context: &C, rng: &mut dyn rand::RngCore) -> Result<A, BanditError> {
        self.policy
            .select(&self.arms, context, rng)
            .ok_or(BanditError::NoArmsAvailable)
    }

    /// Get expected rewards
    pub fn predict_expectations(&self, context: &C) -> HashMap<A, f64> {
        self.policy.expectations(&self.arms, context)
    }

    // Helper methods
    fn validate_decisions(&self, decisions: &[A]) -> Result<(), BanditError> {
        for decision in decisions {
            if !self.arms.contains(decision) {
                return Err(BanditError::ArmNotFound);
            }
        }
        Ok(())
    }

    /// Get the available arms
    pub fn arms(&self) -> &IndexSet<A> {
        &self.arms
    }

    /// Check if an arm exists
    pub fn has_arm(&self, arm: &A) -> bool {
        self.arms.contains(arm)
    }

    /// Add a new arm
    pub fn add_arm(&mut self, arm: A) -> Result<(), BanditError> {
        if self.arms.contains(&arm) {
            return Err(BanditError::ArmAlreadyExists);
        }
        self.arms.insert(arm);
        Ok(())
    }

    /// Remove an arm
    pub fn remove_arm(&mut self, arm: &A) -> Result<(), BanditError> {
        if !self.arms.shift_remove(arm) {
            return Err(BanditError::ArmNotFound);
        }
        self.policy.reset_arm(arm);
        Ok(())
    }

    /// Reset the policy
    pub fn reset(&mut self) {
        self.policy.reset();
    }
}

// Convenience methods for non-contextual policies
impl<A, P> Bandit<A, P, ()>
where
    A: Clone + Eq + Hash,
    P: Policy<A, ()>,
{
    /// Convenience fit method for non-contextual policies
    pub fn fit_simple(&mut self, decisions: &[A], rewards: &[f64]) -> Result<(), BanditError> {
        self.fit(decisions, &(), rewards)
    }

    /// Convenience predict method for non-contextual policies  
    pub fn predict_simple(&self, rng: &mut dyn rand::RngCore) -> Result<A, BanditError> {
        self.predict(&(), rng)
    }

    /// Convenience method to get expectations for non-contextual policies
    pub fn predict_expectations_simple(&self) -> HashMap<A, f64> {
        self.predict_expectations(&())
    }
}

// Special batch method for LinUCB bandits with multiple contexts
impl<A, C> Bandit<A, crate::policies::LinUcb<A>, C>
where
    A: Clone + Eq + Hash,
    C: AsRef<[f64]>,
{
    /// Fit with multiple contexts (one per decision)
    pub fn fit_batch<T: AsRef<[f64]>>(
        &mut self,
        decisions: &[A],
        contexts: &[T],
        rewards: &[f64],
    ) -> Result<(), BanditError> {
        if decisions.len() != contexts.len() || decisions.len() != rewards.len() {
            return Err(BanditError::DimensionMismatch {
                message: format!(
                    "Mismatched dimensions: decisions={}, contexts={}, rewards={}",
                    decisions.len(),
                    contexts.len(),
                    rewards.len()
                ),
            });
        }

        self.validate_decisions(decisions)?;

        // Update each decision with its corresponding context
        for ((decision, ctx), reward) in decisions.iter().zip(contexts).zip(rewards) {
            self.policy.update(decision, ctx, *reward);
        }
        Ok(())
    }
}

// Convenience constructors for common policies
impl<A> Bandit<A, crate::policies::EpsilonGreedy<A>, ()>
where
    A: Clone + Eq + Hash,
{
    /// Create an epsilon-greedy bandit
    pub fn epsilon_greedy<I>(arms: I, epsilon: f64) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        let policy = crate::policies::EpsilonGreedy::new(epsilon);
        Self::new(arms, policy)
    }
}

impl<A> Bandit<A, crate::policies::Random, ()>
where
    A: Clone + Eq + Hash,
{
    /// Create a random bandit
    pub fn random<I>(arms: I) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        let policy = crate::policies::Random;
        Self::new(arms, policy)
    }
}

impl<A> Bandit<A, crate::policies::Ucb<A>, ()>
where
    A: Clone + Eq + Hash,
{
    /// Create a UCB1 bandit
    pub fn ucb<I>(arms: I, alpha: f64) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        let policy = crate::policies::Ucb::new(alpha);
        Self::new(arms, policy)
    }
}

impl<A> Bandit<A, crate::policies::ThompsonSampling<A>, ()>
where
    A: Clone + Eq + Hash,
{
    /// Create a Thompson Sampling bandit
    pub fn thompson_sampling<I>(arms: I) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        let policy = crate::policies::ThompsonSampling::new();
        Self::new(arms, policy)
    }
}

impl<A, C> Bandit<A, crate::policies::LinUcb<A>, C>
where
    A: Clone + Eq + Hash,
    C: AsRef<[f64]>,
{
    /// Create a LinUCB bandit
    pub fn linucb<I>(
        arms: I,
        alpha: f64,
        l2_lambda: f64,
        num_features: usize,
    ) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        let policy = crate::policies::LinUcb::new(alpha, l2_lambda, num_features);
        Self::new(arms, policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_context_free_bandit() {
        let mut bandit = Bandit::epsilon_greedy(vec!["a", "b", "c"], 0.1).unwrap();

        // Train without context using convenience method
        bandit.fit_simple(&["a", "b"], &[1.0, 0.5]).unwrap();

        // Predict without context using convenience method
        let mut rng = StdRng::seed_from_u64(42);
        let choice = bandit.predict_simple(&mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_contextual_bandit() {
        let mut bandit = Bandit::linucb(vec![1, 2, 3], 1.0, 1.0, 2).unwrap();

        // Train with multiple contexts (batch)
        let contexts = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        bandit.fit_batch(&[1, 2], &contexts, &[1.0, 0.5]).unwrap();

        // Predict with single context
        let mut rng = StdRng::seed_from_u64(42);
        let context = vec![0.5, 0.5];
        let choice = bandit.predict(&context, &mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_context_free_can_use_simple_api() {
        let mut bandit = Bandit::epsilon_greedy(vec!["a", "b"], 0.1).unwrap();

        // Context-free bandit uses simple API
        bandit.fit_simple(&["a"], &[1.0]).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let choice = bandit.predict_simple(&mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_contextual_requires_context() {
        let mut bandit = Bandit::linucb(vec![1, 2], 1.0, 1.0, 2).unwrap();

        // Must provide context for contextual bandits - single update
        let context = vec![1.0, 0.0];
        bandit.fit(&[1], &context, &[1.0]).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let choice = bandit.predict(&context, &mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_builder_pattern() {
        // Builder pattern for complex bandits
        let bandit = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(crate::policies::EpsilonGreedy::new(0.1))
            .build()
            .unwrap();

        assert_eq!(bandit.arms().len(), 3);
    }
}

/// Builder for creating bandits
pub struct BanditBuilder<A, P, C = ()> {
    arms: Option<Vec<A>>,
    policy: Option<P>,
    _phantom: PhantomData<fn() -> C>,
}

impl<A, P, C> Default for BanditBuilder<A, P, C> {
    fn default() -> Self {
        Self {
            arms: None,
            policy: None,
            _phantom: PhantomData,
        }
    }
}

impl<A, P, C> BanditBuilder<A, P, C> {
    /// Set the arms
    #[must_use = "builder method returns a new builder that should be used"]
    pub fn arms(mut self, arms: Vec<A>) -> Self {
        self.arms = Some(arms);
        self
    }

    /// Set the policy
    #[must_use = "builder method returns a new builder that should be used"]
    pub fn policy(mut self, policy: P) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Build the bandit
    #[must_use = "the builder should be used to create a bandit"]
    pub fn build(self) -> Result<Bandit<A, P, C>, BanditError>
    where
        A: Clone + Eq + Hash,
        P: Policy<A, C>,
    {
        let arms = self.arms.ok_or(BanditError::BuilderError {
            message: "Arms not specified".to_string(),
        })?;
        let policy = self.policy.ok_or(BanditError::BuilderError {
            message: "Policy not specified".to_string(),
        })?;
        Bandit::new(arms, policy)
    }
}

impl<A, P, C> Bandit<A, P, C> {
    /// Create a new builder
    pub fn builder() -> BanditBuilder<A, P, C> {
        BanditBuilder::default()
    }
}
