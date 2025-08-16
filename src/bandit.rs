use crate::error::BanditError;
use crate::policies::{LinUcb, Policy};
use indexmap::IndexSet;
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// A multi-armed bandit with a specific policy
///
/// The `Bandit` struct maintains a set of arms and uses a policy to make decisions.
/// It provides a type-safe, generic interface that works with any hashable arm type
/// and any policy implementation.
#[derive(Clone, Debug)]
pub struct Bandit<A, P, C = ()> {
    arms: IndexSet<A>,
    policy: P,
    _phantom: PhantomData<fn() -> C>,
}

impl<A, P, C> Bandit<A, P, C>
where
    A: Clone + Eq + Hash,
    P: Policy<A, C>,
{
    /// Creates a new bandit with the given arms and policy
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

    /// Gets the available arms
    pub fn arms(&self) -> &IndexSet<A> {
        &self.arms
    }

    /// Check if an arm exists in the bandit
    pub fn has_arm(&self, arm: &A) -> bool {
        self.arms.contains(arm)
    }

    /// Gets a reference to the policy
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Gets a mutable reference to the policy
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Add a new arm to the bandit
    pub fn add_arm(&mut self, arm: A) -> Result<(), BanditError> {
        if self.arms.contains(&arm) {
            return Err(BanditError::ArmAlreadyExists {
                arm: "arm".to_string(), // Generic error message without Debug
            });
        }
        self.arms.insert(arm.clone());
        self.policy.reset_arm(&arm);
        Ok(())
    }

    /// Remove an arm from the bandit
    pub fn remove_arm(&mut self, arm: &A) -> Result<(), BanditError> {
        if !self.arms.contains(arm) {
            return Err(BanditError::ArmNotFound {
                arm: "arm".to_string(), // Generic error message without Debug
            });
        }
        self.arms.shift_remove(arm);
        Ok(())
    }

    /// Validate that all decisions correspond to valid arms
    fn validate_decisions(&self, decisions: &[A]) -> Result<(), BanditError> {
        for decision in decisions {
            if !self.arms.contains(decision) {
                return Err(BanditError::ArmNotFound {
                    arm: "decision".to_string(), // Generic error message without Debug
                });
            }
        }
        Ok(())
    }
}

// Convenience methods for non-contextual bandits
impl<A, P> Bandit<A, P, ()>
where
    A: Clone + Eq + Hash,
    P: Policy<A, ()>,
{
    /// Fit without context for non-contextual bandits
    pub fn fit_simple(&mut self, decisions: &[A], rewards: &[f64]) -> Result<(), BanditError> {
        self.fit(decisions, &(), rewards)
    }

    /// Predict without context for non-contextual bandits
    pub fn predict_simple(&self, rng: &mut dyn rand::RngCore) -> Result<A, BanditError> {
        self.predict(&(), rng)
    }

    /// Get expectations without context for non-contextual bandits
    pub fn predict_expectations_simple(&self) -> HashMap<A, f64> {
        self.predict_expectations(&())
    }
}

// Batch operations for LinUCB
impl<A, C> Bandit<A, LinUcb<A>, C>
where
    A: Clone + Eq + Hash,
    C: AsRef<[f64]>,
{
    /// Fit LinUCB with multiple contexts
    pub fn fit_batch(
        &mut self,
        decisions: &[A],
        contexts: &[C],
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
        Self::new(arms, crate::policies::EpsilonGreedy::new(epsilon))
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
        Self::new(arms, crate::policies::Random)
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
        Self::new(arms, crate::policies::Ucb::new(alpha))
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
        Self::new(arms, crate::policies::ThompsonSampling::new())
    }
}

impl<A> Bandit<A, crate::policies::LinUcb<A>, Vec<f64>>
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
        Self::new(
            arms,
            crate::policies::LinUcb::new(alpha, l2_lambda, num_features),
        )
    }
}

/// Builder for creating bandits with a fluent API
pub struct BanditBuilder<A, P, C = ()> {
    arms: Option<IndexSet<A>>,
    policy: Option<P>,
    _phantom: PhantomData<C>,
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

impl<A, P, C> BanditBuilder<A, P, C>
where
    A: Clone + Eq + Hash,
    P: Policy<A, C>,
{
    /// Set the arms for the bandit
    pub fn arms<I>(mut self, arms: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        self.arms = Some(arms.into_iter().collect());
        self
    }

    /// Set the policy for the bandit
    pub fn policy(mut self, policy: P) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Build the bandit
    pub fn build(self) -> Result<Bandit<A, P, C>, BanditError> {
        let arms = self.arms.ok_or(BanditError::BuilderError {
            message: "Arms not specified".into(),
        })?;

        let policy = self.policy.ok_or(BanditError::BuilderError {
            message: "Policy not specified".into(),
        })?;

        if arms.is_empty() {
            return Err(BanditError::NoArmsAvailable);
        }

        Ok(Bandit {
            arms,
            policy,
            _phantom: PhantomData,
        })
    }
}

impl<A, P, C> Bandit<A, P, C> {
    /// Create a new builder for constructing a bandit
    pub fn builder() -> BanditBuilder<A, P, C> {
        BanditBuilder::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::{EpsilonGreedy, Random};
    use rand::SeedableRng;

    #[test]
    fn test_bandit_creation() {
        let bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();
        assert_eq!(bandit.arms().len(), 3);
    }

    #[test]
    fn test_bandit_builder() {
        let bandit = Bandit::<i32, Random, ()>::builder()
            .arms(vec![1, 2, 3])
            .policy(Random)
            .build()
            .unwrap();
        assert_eq!(bandit.arms().len(), 3);
    }

    #[test]
    fn test_non_contextual_convenience_methods() {
        let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.1).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Use simple methods that don't require &()
        bandit.fit_simple(&[1, 2], &[1.0, 0.5]).unwrap();
        let choice = bandit.predict_simple(&mut rng).unwrap();
        assert!(bandit.arms().contains(&choice));

        let expectations = bandit.predict_expectations_simple();
        assert_eq!(expectations.len(), 3);
    }

    #[test]
    fn test_add_remove_arms() {
        let mut bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();

        // Add a new arm
        bandit.add_arm(4).unwrap();
        assert_eq!(bandit.arms().len(), 4);

        // Try to add duplicate
        assert!(bandit.add_arm(4).is_err());

        // Remove an arm
        bandit.remove_arm(&2).unwrap();
        assert_eq!(bandit.arms().len(), 3);
        assert!(!bandit.arms().contains(&2));

        // Try to remove non-existent arm
        assert!(bandit.remove_arm(&10).is_err());
    }
}
