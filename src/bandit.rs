//! Unified bandit implementation that handles both contextual and context-free cases.

use crate::error::BanditError;
use crate::policies::Policy;
use indexmap::IndexSet;
use std::collections::HashMap;
use std::hash::Hash;

/// Bandit that handles both contextual and context-free scenarios
pub struct Bandit<A, P> {
    /// Available arms
    arms: IndexSet<A>,
    /// The policy (now unified)
    policy: P,
}

impl<A, P> Bandit<A, P>
where
    A: Clone + Eq + Hash,
    P: Policy<A>,
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

        Ok(Self { arms, policy })
    }

    /// Fit the bandit with training data (context-free version)
    pub fn fit(&mut self, decisions: &[A], rewards: &[f64]) -> Result<(), BanditError> {
        if self.policy.requires_context() {
            return Err(BanditError::DimensionMismatch {
                message: "This bandit requires context features. Use fit_with_context() instead."
                    .to_string(),
            });
        }

        self.validate_decisions(decisions)?;
        self.policy.update(decisions, rewards);
        Ok(())
    }

    /// Incrementally fit the bandit (alias for fit in context-free case)
    pub fn partial_fit(&mut self, decisions: &[A], rewards: &[f64]) -> Result<(), BanditError> {
        self.fit(decisions, rewards)
    }

    /// Fit the bandit with training data (contextual version)
    pub fn fit_with_context(
        &mut self,
        decisions: &[A],
        contexts: &[Vec<f64>],
        rewards: &[f64],
    ) -> Result<(), BanditError> {
        if !self.policy.requires_context() {
            // Allow using context API with context-free policies (just ignore context)
            return self.fit(decisions, rewards);
        }

        self.validate_decisions(decisions)?;
        self.validate_contexts(contexts)?;

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

        self.policy
            .update_with_context(decisions, Some(contexts), rewards)?;
        Ok(())
    }

    /// Make a prediction (context-free version)
    pub fn predict(&self, rng: &mut dyn rand::RngCore) -> Result<A, BanditError> {
        if self.policy.requires_context() {
            return Err(BanditError::DimensionMismatch {
                message:
                    "This bandit requires context features. Use predict_with_context() instead."
                        .to_string(),
            });
        }

        self.policy
            .select(&self.arms, rng)
            .ok_or(BanditError::NoArmsAvailable)
    }

    /// Make a prediction (contextual version)
    pub fn predict_with_context(
        &self,
        context: &[f64],
        rng: &mut dyn rand::RngCore,
    ) -> Result<A, BanditError> {
        if !self.policy.requires_context() {
            // Allow using context API with context-free policies (just ignore context)
            return self.predict(rng);
        }

        if context.len() != self.policy.num_features() {
            return Err(BanditError::DimensionMismatch {
                message: format!(
                    "Context has {} features, expected {}",
                    context.len(),
                    self.policy.num_features()
                ),
            });
        }

        self.policy
            .select_with_context(&self.arms, Some(context), rng)?
            .ok_or(BanditError::NoArmsAvailable)
    }

    /// Get expected rewards (context-free version)
    pub fn predict_expectations(&self) -> HashMap<A, f64> {
        self.policy.expectations(&self.arms)
    }

    /// Get expected rewards (contextual version)
    pub fn predict_expectations_with_context(
        &self,
        context: &[f64],
    ) -> Result<HashMap<A, f64>, BanditError> {
        if !self.policy.requires_context() {
            // Allow using context API with context-free policies (just ignore context)
            return Ok(self.predict_expectations());
        }

        if context.len() != self.policy.num_features() {
            return Err(BanditError::DimensionMismatch {
                message: format!(
                    "Context has {} features, expected {}",
                    context.len(),
                    self.policy.num_features()
                ),
            });
        }

        Ok(self
            .policy
            .expectations_with_context(&self.arms, Some(context))?)
    }

    /// Check if this bandit requires context
    pub fn requires_context(&self) -> bool {
        self.policy.requires_context()
    }

    /// Get the number of features expected
    pub fn num_features(&self) -> usize {
        self.policy.num_features()
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

    fn validate_contexts(&self, contexts: &[Vec<f64>]) -> Result<(), BanditError> {
        let expected_features = self.policy.num_features();
        for context in contexts {
            if context.len() != expected_features {
                return Err(BanditError::DimensionMismatch {
                    message: format!(
                        "Context has {} features, expected {}",
                        context.len(),
                        expected_features
                    ),
                });
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

// Convenience constructors for common policies
impl<A> Bandit<A, crate::policies::EpsilonGreedy<A>>
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

impl<A> Bandit<A, crate::policies::Random>
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

impl<A> Bandit<A, crate::policies::Ucb<A>>
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

impl<A> Bandit<A, crate::policies::ThompsonSampling<A>>
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

impl<A> Bandit<A, crate::policies::LinUcb<A>>
where
    A: Clone + Eq + Hash,
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

        assert!(!bandit.requires_context());
        assert_eq!(bandit.num_features(), 0);

        // Train without context
        bandit.fit(&["a", "b"], &[1.0, 0.5]).unwrap();

        // Predict without context
        let mut rng = StdRng::seed_from_u64(42);
        let choice = bandit.predict(&mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_contextual_bandit() {
        let mut bandit = Bandit::linucb(vec![1, 2, 3], 1.0, 1.0, 2).unwrap();

        assert!(bandit.requires_context());
        assert_eq!(bandit.num_features(), 2);

        // Train with context
        let contexts = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        bandit
            .fit_with_context(&[1, 2], &contexts, &[1.0, 0.5])
            .unwrap();

        // Predict with context
        let mut rng = StdRng::seed_from_u64(42);
        let choice = bandit.predict_with_context(&[0.5, 0.5], &mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_context_free_can_use_context_api() {
        let mut bandit = Bandit::epsilon_greedy(vec!["a", "b"], 0.1).unwrap();

        // Context-free bandit can use context API (context is ignored)
        let contexts = vec![vec![1.0, 0.0]];
        bandit.fit_with_context(&["a"], &contexts, &[1.0]).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let choice = bandit.predict_with_context(&[0.5, 0.5], &mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));
    }

    #[test]
    fn test_contextual_requires_context() {
        let mut bandit = Bandit::linucb(vec![1, 2], 1.0, 1.0, 2).unwrap();

        // Trying to use context-free API should fail
        assert!(bandit.fit(&[1], &[1.0]).is_err());

        let mut rng = StdRng::seed_from_u64(42);
        assert!(bandit.predict(&mut rng).is_err());
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
pub struct BanditBuilder<A, P> {
    arms: Option<Vec<A>>,
    policy: Option<P>,
}

impl<A, P> Default for BanditBuilder<A, P> {
    fn default() -> Self {
        Self {
            arms: None,
            policy: None,
        }
    }
}

impl<A, P> BanditBuilder<A, P> {
    /// Set the arms
    pub fn arms(mut self, arms: Vec<A>) -> Self {
        self.arms = Some(arms);
        self
    }

    /// Set the policy
    pub fn policy(mut self, policy: P) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Build the bandit
    pub fn build(self) -> Result<Bandit<A, P>, BanditError>
    where
        A: Clone + Eq + Hash,
        P: Policy<A>,
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

impl<A, P> Bandit<A, P> {
    /// Create a new builder
    pub fn builder() -> BanditBuilder<A, P> {
        BanditBuilder::default()
    }
}
