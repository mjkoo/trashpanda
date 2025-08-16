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

    /// Fit the bandit with training data
    ///
    /// This method resets the policy and trains it with the provided data.
    /// For incremental updates without resetting, use [`partial_fit`](Self::partial_fit).
    /// For non-contextual bandits, use [`fit_simple`](Self::fit_simple) instead.
    ///
    /// # Arguments
    /// - `decisions`: The arms that were selected
    /// - `context`: The context when decisions were made (same for all)
    /// - `rewards`: The observed rewards
    ///
    /// # Why `C: Copy`?
    ///
    /// The `Copy` bound is required because the same context needs to be passed to
    /// multiple update calls in the loop. This design supports both:
    /// - Trivial types like `()` for non-contextual bandits
    /// - References like `&[f64]` for contextual bandits
    pub fn fit(&mut self, decisions: &[A], context: C, rewards: &[f64]) -> Result<(), BanditError>
    where
        C: Copy,
    {
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

        // Reset the policy state before training
        self.policy.reset();

        // Train with the provided data
        for (decision, reward) in decisions.iter().zip(rewards) {
            self.policy.update(decision, context, *reward);
        }
        Ok(())
    }

    /// Incrementally fit the bandit with new observations
    ///
    /// This method updates the policy without resetting its state, adding to
    /// existing knowledge. For training from scratch, use [`fit`](Self::fit).
    ///
    /// # Arguments
    /// - `decisions`: The arms that were selected
    /// - `context`: The context when decisions were made (same for all)
    /// - `rewards`: The observed rewards
    ///
    /// # Difference from `fit`
    /// - `fit`: Resets the policy state before training (replaces knowledge)
    /// - `partial_fit`: Adds to existing state (incremental learning)
    pub fn partial_fit(
        &mut self,
        decisions: &[A],
        context: C,
        rewards: &[f64],
    ) -> Result<(), BanditError>
    where
        C: Copy,
    {
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

        // Update without resetting (incremental)
        for (decision, reward) in decisions.iter().zip(rewards) {
            self.policy.update(decision, context, *reward);
        }
        Ok(())
    }

    /// Make a prediction
    pub fn predict(&self, context: C, rng: &mut dyn rand::RngCore) -> Result<A, BanditError> {
        self.policy
            .select(&self.arms, context, rng)
            .ok_or(BanditError::NoArmsAvailable)
    }

    /// Get expected rewards
    pub fn predict_expectations(&self, context: C, rng: &mut dyn rand::RngCore) -> HashMap<A, f64> {
        self.policy.expectations(&self.arms, context, rng)
    }

    /// Make predictions for multiple contexts
    ///
    /// Returns a vector of predictions, one for each context provided.
    /// Each prediction uses the same arms but potentially different contexts.
    ///
    /// # Arguments
    /// * `contexts` - Slice of contexts to make predictions for
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of predicted arms, one for each context
    pub fn predict_batch<I>(
        &self,
        contexts: I,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<A>, BanditError>
    where
        I: IntoIterator<Item = C>,
        C: Copy,
    {
        let mut predictions = Vec::new();
        for context in contexts {
            let prediction = self.predict(context, rng)?;
            predictions.push(prediction);
        }
        Ok(predictions)
    }

    /// Get expected rewards for multiple contexts
    ///
    /// Returns a vector of expectation maps, one for each context provided.
    ///
    /// # Arguments
    /// * `contexts` - Slice of contexts to get expectations for
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of HashMaps containing arm->reward expectations
    pub fn predict_expectations_batch<I>(
        &self,
        contexts: I,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<HashMap<A, f64>>
    where
        I: IntoIterator<Item = C>,
        C: Copy,
    {
        let mut expectations = Vec::new();
        for context in contexts {
            let expectation = self.predict_expectations(context, rng);
            expectations.push(expectation);
        }
        expectations
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
    ///
    /// Resets the policy state and trains from scratch.
    pub fn fit_simple(&mut self, decisions: &[A], rewards: &[f64]) -> Result<(), BanditError> {
        self.fit(decisions, (), rewards)
    }

    /// Incrementally fit without context for non-contextual bandits
    ///
    /// Adds to existing knowledge without resetting.
    pub fn partial_fit_simple(
        &mut self,
        decisions: &[A],
        rewards: &[f64],
    ) -> Result<(), BanditError> {
        self.partial_fit(decisions, (), rewards)
    }

    /// Predict without context for non-contextual bandits
    pub fn predict_simple(&self, rng: &mut dyn rand::RngCore) -> Result<A, BanditError> {
        self.predict((), rng)
    }

    /// Get expectations without context for non-contextual bandits
    pub fn predict_expectations_simple(&self, rng: &mut dyn rand::RngCore) -> HashMap<A, f64> {
        self.predict_expectations((), rng)
    }

    /// Make batch predictions without context for non-contextual bandits
    ///
    /// This is equivalent to calling `predict_simple()` multiple times.
    /// Useful when you need multiple independent predictions from the same bandit state.
    ///
    /// # Arguments
    /// * `count` - Number of predictions to make
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of predicted arms
    pub fn predict_batch_simple(
        &self,
        count: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<A>, BanditError> {
        let contexts = vec![(); count];
        self.predict_batch(contexts, rng)
    }

    /// Get batch expectations without context for non-contextual bandits
    ///
    /// This is equivalent to calling `predict_expectations_simple()` multiple times.
    /// Useful when you need multiple independent expectation calculations.
    ///
    /// # Arguments
    /// * `count` - Number of expectation calculations to make
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of HashMap containing arm->reward expectations
    pub fn predict_expectations_batch_simple(
        &self,
        count: usize,
        rng: &mut dyn rand::RngCore,
    ) -> Vec<HashMap<A, f64>> {
        let contexts = vec![(); count];
        self.predict_expectations_batch(contexts, rng)
    }
}

// Batch operations for LinUCB
impl<A> Bandit<A, LinUcb<A>, &[f64]>
where
    A: Clone + Eq + Hash,
{
    /// Fit LinUCB with multiple contexts
    pub fn fit_batch<C>(
        &mut self,
        decisions: &[A],
        contexts: &[C],
        rewards: &[f64],
    ) -> Result<(), BanditError>
    where
        C: AsRef<[f64]>,
    {
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
            self.policy.update(decision, ctx.as_ref(), *reward);
        }
        Ok(())
    }

    /// Make batch predictions for LinUCB with different contexts
    ///
    /// This is more efficient than calling `predict()` multiple times when you have
    /// many different contexts to evaluate.
    ///
    /// # Arguments
    /// * `contexts` - Slice of contexts, each context can be different
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of predicted arms, one for each context
    pub fn predict_batch_contexts<C>(
        &self,
        contexts: &[C],
        rng: &mut dyn rand::RngCore,
    ) -> Result<Vec<A>, BanditError>
    where
        C: AsRef<[f64]>,
    {
        let mut predictions = Vec::with_capacity(contexts.len());
        for context in contexts {
            let prediction = self.predict(context.as_ref(), rng)?;
            predictions.push(prediction);
        }
        Ok(predictions)
    }

    /// Get batch expectations for LinUCB with different contexts
    ///
    /// # Arguments
    /// * `contexts` - Slice of contexts, each context can be different
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of HashMaps containing arm->reward expectations
    pub fn predict_expectations_batch_contexts<C>(
        &self,
        contexts: &[C],
        rng: &mut dyn rand::RngCore,
    ) -> Vec<HashMap<A, f64>>
    where
        C: AsRef<[f64]>,
    {
        let mut expectations = Vec::with_capacity(contexts.len());
        for context in contexts {
            let expectation = self.predict_expectations(context.as_ref(), rng);
            expectations.push(expectation);
        }
        expectations
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

impl<A> Bandit<A, crate::policies::Random<A>, ()>
where
    A: Clone + Eq + Hash,
{
    /// Create a random bandit
    pub fn random<I>(arms: I) -> Result<Self, BanditError>
    where
        I: IntoIterator<Item = A>,
    {
        Self::new(arms, crate::policies::Random::default())
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

impl<A> Bandit<A, crate::policies::LinUcb<A>, &[f64]>
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
    use crate::policies::Random;
    use rand::SeedableRng;

    #[test]
    fn test_bandit_creation() {
        let bandit = Bandit::new(vec![1, 2, 3], Random::default()).unwrap();
        assert_eq!(bandit.arms().len(), 3);
    }

    #[test]
    fn test_bandit_builder() {
        let bandit = Bandit::<i32, Random<i32>, ()>::builder()
            .arms(vec![1, 2, 3])
            .policy(Random::default())
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

        let expectations = bandit.predict_expectations_simple(&mut rng);
        assert_eq!(expectations.len(), 3);
    }

    #[test]
    fn test_partial_fit_method() {
        let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.0).unwrap(); // Pure exploitation

        // Test incremental updates
        bandit.partial_fit(&[1], (), &[1.0]).unwrap();
        bandit.partial_fit(&[2], (), &[0.5]).unwrap();
        bandit.partial_fit(&[1], (), &[0.8]).unwrap();

        // Test batch partial fit
        bandit
            .partial_fit(&[1, 2, 3], (), &[0.9, 0.3, 0.4])
            .unwrap();

        // Test invalid arm
        assert!(bandit.partial_fit(&[4], (), &[1.0]).is_err());

        // Verify it learned (arm 1 should be preferred with exploitation)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let choice = bandit.predict_simple(&mut rng).unwrap();
        assert_eq!(choice, 1); // Should pick arm 1 with highest average reward
    }

    #[test]
    fn test_partial_fit_simple_method() {
        let mut bandit = Bandit::epsilon_greedy(vec!["a", "b", "c"], 0.0).unwrap();

        // Test partial_fit_simple with single observations
        bandit.partial_fit_simple(&["a"], &[1.0]).unwrap();
        bandit.partial_fit_simple(&["b"], &[0.5]).unwrap();
        bandit.partial_fit_simple(&["a"], &[0.8]).unwrap();

        // Test batch update
        bandit
            .partial_fit_simple(&["a", "b", "c"], &[0.9, 0.3, 0.4])
            .unwrap();

        // Test invalid arm
        assert!(bandit.partial_fit_simple(&["d"], &[1.0]).is_err());

        // Verify it learned
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let choice = bandit.predict_simple(&mut rng).unwrap();
        assert_eq!(choice, "a"); // Should pick arm "a" with highest average reward
    }

    #[test]
    fn test_fit_vs_partial_fit() {
        let mut bandit1 = Bandit::epsilon_greedy(vec![1, 2, 3], 0.0).unwrap();
        let mut bandit2 = Bandit::epsilon_greedy(vec![1, 2, 3], 0.0).unwrap();

        // Train bandit1 with fit (should reset each time)
        bandit1.fit(&[1, 1, 2], (), &[1.0, 0.8, 0.5]).unwrap();
        bandit1.fit(&[3, 3], (), &[0.9, 0.95]).unwrap(); // This replaces previous training

        // Train bandit2 with partial_fit (should accumulate)
        bandit2
            .partial_fit(&[1, 1, 2], (), &[1.0, 0.8, 0.5])
            .unwrap();
        bandit2.partial_fit(&[3, 3], (), &[0.9, 0.95]).unwrap(); // This adds to previous training

        // Get expectations
        let mut rng_exp1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng_exp2 = rand::rngs::StdRng::seed_from_u64(42);
        let exp1 = bandit1.predict_expectations_simple(&mut rng_exp1);
        let exp2 = bandit2.predict_expectations_simple(&mut rng_exp2);

        // bandit1 should only have data from the second fit (arms 3)
        // bandit2 should have accumulated data from both partial_fits

        // For bandit1: only arm 3 has data (0.9 + 0.95) / 2 = 0.925
        assert_eq!(exp1.get(&1), Some(&0.0)); // No data after reset
        assert_eq!(exp1.get(&2), Some(&0.0)); // No data after reset
        assert!(*exp1.get(&3).unwrap() > 0.9); // Has recent data

        // For bandit2: all arms have accumulated data
        assert!(*exp2.get(&1).unwrap() > 0.0); // Has data from first partial_fit
        assert!(*exp2.get(&2).unwrap() > 0.0); // Has data from first partial_fit
        assert!(*exp2.get(&3).unwrap() > 0.0); // Has data from second partial_fit
    }

    #[test]
    fn test_add_remove_arms() {
        let mut bandit = Bandit::new(vec![1, 2, 3], Random::default()).unwrap();

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

    #[test]
    fn test_predict_batch() {
        let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.0).unwrap(); // Pure exploitation
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Train the bandit
        bandit.fit(&[1, 1, 2], (), &[1.0, 0.9, 0.5]).unwrap();

        // Test batch prediction with same context
        let contexts = vec![(), (), ()];
        let predictions = bandit.predict_batch(contexts, &mut rng).unwrap();
        assert_eq!(predictions.len(), 3);
        // With epsilon=0, should always pick arm 1 (highest reward)
        for prediction in predictions {
            assert_eq!(prediction, 1);
        }

        // Test empty batch
        let empty_contexts: Vec<()> = vec![];
        let empty_predictions = bandit.predict_batch(empty_contexts, &mut rng).unwrap();
        assert_eq!(empty_predictions.len(), 0);
    }

    #[test]
    fn test_predict_expectations_batch() {
        let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.1).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Train the bandit
        bandit.fit(&[1, 1, 2], (), &[1.0, 0.9, 0.5]).unwrap();

        // Test batch expectations
        let contexts = vec![(), (), ()];
        let expectations_batch = bandit.predict_expectations_batch(contexts, &mut rng);
        assert_eq!(expectations_batch.len(), 3);

        // All should be identical since context is the same
        for expectations in &expectations_batch {
            assert_eq!(expectations.len(), 3); // 3 arms
            assert!(expectations.contains_key(&1));
            assert!(expectations.contains_key(&2));
            assert!(expectations.contains_key(&3));
        }

        // Expectations should be identical for same context
        assert_eq!(expectations_batch[0], expectations_batch[1]);
        assert_eq!(expectations_batch[1], expectations_batch[2]);
    }

    #[test]
    fn test_predict_batch_simple() {
        let mut bandit = Bandit::epsilon_greedy(vec!["a", "b", "c"], 0.0).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Train the bandit
        bandit
            .fit_simple(&["a", "a", "b"], &[1.0, 0.9, 0.5])
            .unwrap();

        // Test batch prediction simple
        let predictions = bandit.predict_batch_simple(5, &mut rng).unwrap();
        assert_eq!(predictions.len(), 5);
        // With epsilon=0, should always pick "a" (highest reward)
        for prediction in predictions {
            assert_eq!(prediction, "a");
        }

        // Test zero count
        let zero_predictions = bandit.predict_batch_simple(0, &mut rng).unwrap();
        assert_eq!(zero_predictions.len(), 0);
    }

    #[test]
    fn test_predict_expectations_batch_simple() {
        let mut bandit = Bandit::epsilon_greedy(vec!["x", "y", "z"], 0.1).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Train the bandit
        bandit
            .fit_simple(&["x", "x", "y"], &[1.0, 0.9, 0.5])
            .unwrap();

        // Test batch expectations simple
        let expectations_batch = bandit.predict_expectations_batch_simple(3, &mut rng);
        assert_eq!(expectations_batch.len(), 3);

        // All should be identical since no context
        for expectations in &expectations_batch {
            assert_eq!(expectations.len(), 3); // 3 arms
            assert!(expectations.contains_key(&"x"));
            assert!(expectations.contains_key(&"y"));
            assert!(expectations.contains_key(&"z"));
        }

        // All expectations should be identical
        assert_eq!(expectations_batch[0], expectations_batch[1]);
        assert_eq!(expectations_batch[1], expectations_batch[2]);
    }

    #[test]
    fn test_linucb_batch_contexts() {
        let mut bandit = Bandit::linucb(vec![1, 2], 1.0, 1.0, 2).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Train with different contexts
        let contexts = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        bandit
            .fit_batch(&[1, 2, 1], &contexts, &[1.0, 0.5, 0.8])
            .unwrap();

        // Test batch prediction with different contexts
        let test_contexts = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let predictions = bandit
            .predict_batch_contexts(&test_contexts, &mut rng)
            .unwrap();
        assert_eq!(predictions.len(), 2);

        // Test batch expectations with different contexts
        let expectations_batch =
            bandit.predict_expectations_batch_contexts(&test_contexts, &mut rng);
        assert_eq!(expectations_batch.len(), 2);

        for expectations in &expectations_batch {
            assert_eq!(expectations.len(), 2); // 2 arms
            assert!(expectations.contains_key(&1));
            assert!(expectations.contains_key(&2));
        }

        // Different contexts should potentially give different expectations
        // (though this is not guaranteed depending on the training data)
    }
}
