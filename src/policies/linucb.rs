use crate::error::{PolicyError, PolicyResult};
use crate::policies::Policy;
use crate::ridge::RidgeRegression;
use indexmap::IndexSet;
use rand::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

/// Linear Upper Confidence Bound (LinUCB) policy for contextual bandits
///
/// LinUCB uses ridge regression to model the expected reward for each arm
/// given context features, and adds an upper confidence bound for exploration.
///
/// The algorithm maintains a separate ridge regression model for each arm.
#[derive(Debug, Clone)]
pub struct LinUcb<A> {
    /// Exploration parameter (controls confidence bound width)
    alpha: f64,
    /// Regularization parameter for ridge regression
    l2_lambda: f64,
    /// Number of context features
    num_features: usize,
    /// Ridge regression model for each arm
    arm_models: HashMap<A, RidgeRegression>,
}

impl<A> LinUcb<A>
where
    A: Clone + Eq + Hash,
{
    /// Create a new LinUCB policy
    ///
    /// # Arguments
    /// * `alpha` - Exploration parameter (typically between 0.1 and 2.0)
    /// * `l2_lambda` - L2 regularization parameter (typically 1.0)
    /// * `num_features` - Number of context features
    #[must_use]
    pub fn new(alpha: f64, l2_lambda: f64, num_features: usize) -> Self {
        Self {
            alpha,
            l2_lambda,
            num_features,
            arm_models: HashMap::new(),
        }
    }

    /// Get or create a model for an arm
    fn get_or_create_model(&mut self, arm: &A) -> &mut RidgeRegression {
        self.arm_models
            .entry(arm.clone())
            .or_insert_with(|| RidgeRegression::new(self.num_features, self.l2_lambda))
    }

    /// Calculate the upper confidence bound for an arm given context
    fn calculate_ucb(&self, arm: &A, context: &[f64]) -> f64 {
        if let Some(model) = self.arm_models.get(arm) {
            let prediction = model.predict(context);
            let variance = model.variance(context);
            prediction + self.alpha * variance
        } else {
            // Uninitalized arm gets maximum UCB for exploration
            f64::INFINITY
        }
    }
}

impl<A> Policy<A> for LinUcb<A>
where
    A: Clone + Eq + Hash,
{
    // LinUCB is a contextual policy, these methods return errors
    fn update(&mut self, _decisions: &[A], _rewards: &[f64]) {
        // LinUCB requires context, this method should not be called
    }

    fn select(&self, _arms: &IndexSet<A>, _rng: &mut dyn rand::RngCore) -> Option<A> {
        // LinUCB requires context, this method should not be called
        None
    }

    fn expectations(&self, _arms: &IndexSet<A>) -> HashMap<A, f64> {
        // LinUCB requires context, this method should not be called
        HashMap::new()
    }

    // LinUCB implements the contextual methods
    fn update_with_context(
        &mut self,
        decisions: &[A],
        contexts: Option<&[Vec<f64>]>,
        rewards: &[f64],
    ) -> PolicyResult<()> {
        let contexts = contexts.ok_or(PolicyError::ContextRequired)?;
        for ((arm, context), reward) in decisions.iter().zip(contexts).zip(rewards) {
            let model = self.get_or_create_model(arm);
            model.fit(context, *reward);
        }
        Ok(())
    }

    fn select_with_context(
        &self,
        arms: &IndexSet<A>,
        context: Option<&[f64]>,
        rng: &mut dyn rand::RngCore,
    ) -> PolicyResult<Option<A>> {
        let context = context.ok_or(PolicyError::ContextRequired)?;

        if arms.is_empty() {
            return Ok(None);
        }

        // Calculate UCB for each arm
        let mut best_arms = Vec::new();
        let mut best_ucb = f64::NEG_INFINITY;

        for arm in arms {
            let ucb = self.calculate_ucb(arm, context);

            if (ucb - best_ucb).abs() < 1e-10 {
                // Tie - add to best arms
                best_arms.push(arm.clone());
            } else if ucb > best_ucb {
                // New best
                best_ucb = ucb;
                best_arms.clear();
                best_arms.push(arm.clone());
            }
        }

        // Break ties randomly
        Ok(best_arms.choose(rng).cloned())
    }

    fn expectations_with_context(
        &self,
        arms: &IndexSet<A>,
        context: Option<&[f64]>,
    ) -> PolicyResult<HashMap<A, f64>> {
        let context = context.ok_or(PolicyError::ContextRequired)?;

        let mut expectations = HashMap::new();

        for arm in arms {
            if let Some(model) = self.arm_models.get(arm) {
                expectations.insert(arm.clone(), model.predict(context));
            } else {
                // Uninitialized arms get zero expectation
                expectations.insert(arm.clone(), 0.0);
            }
        }

        Ok(expectations)
    }

    fn reset_arm(&mut self, arm: &A) {
        if let Some(model) = self.arm_models.get_mut(arm) {
            model.reset();
        }
    }

    fn reset(&mut self) {
        self.arm_models.clear();
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn num_features(&self) -> usize {
        self.num_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_linucb_creation() {
        let policy: LinUcb<&str> = LinUcb::new(1.0, 1.0, 3);
        assert_eq!(policy.num_features(), 3);
    }

    #[test]
    fn test_linucb_select_explores_uninitialized() {
        let policy: LinUcb<i32> = LinUcb::new(1.0, 1.0, 2);
        let arms = IndexSet::from([1, 2, 3]);
        let context = vec![0.5, 0.5];
        let mut rng = StdRng::seed_from_u64(42);

        // Should select from uninitialized arms
        let selected = policy
            .select_with_context(&arms, Some(&context), &mut rng)
            .unwrap();
        assert!(selected.is_some());
        assert!(arms.contains(&selected.unwrap()));
    }

    #[test]
    fn test_linucb_update_and_predict() {
        let mut policy: LinUcb<&str> = LinUcb::new(0.5, 1.0, 2);
        let arms = IndexSet::from(["a", "b"]);

        // Train with some data
        let decisions = vec!["a", "a", "b"];
        let contexts = vec![vec![1.0, 0.0], vec![0.8, 0.2], vec![0.0, 1.0]];
        let rewards = vec![1.0, 0.8, 0.2];

        policy
            .update_with_context(&decisions, Some(&contexts), &rewards)
            .unwrap();

        // Check expectations
        let context = vec![1.0, 0.0];
        let expectations = policy
            .expectations_with_context(&arms, Some(&context))
            .unwrap();

        // Arm "a" should have higher expectation for context [1.0, 0.0]
        assert!(expectations[&"a"] > expectations[&"b"]);
    }

    #[test]
    fn test_linucb_reset() {
        let mut policy: LinUcb<i32> = LinUcb::new(1.0, 1.0, 2);

        // Train with some data
        policy
            .update_with_context(&[1], Some(&[vec![1.0, 0.0]]), &[1.0])
            .unwrap();

        // Reset and check that model is cleared
        policy.reset();

        let arms = IndexSet::from([1, 2]);
        let expectations = policy
            .expectations_with_context(&arms, Some(&[1.0, 0.0]))
            .unwrap();

        // After reset, all arms should have zero expectation
        assert_eq!(expectations[&1], 0.0);
        assert_eq!(expectations[&2], 0.0);
    }

    #[test]
    fn test_linucb_exploration() {
        let mut policy: LinUcb<&str> = LinUcb::new(2.0, 1.0, 2); // High alpha for exploration
        let arms = IndexSet::from(["exploit", "explore"]);
        let mut rng = StdRng::seed_from_u64(42);

        // Train "exploit" arm to have high reward
        for _ in 0..10 {
            policy
                .update_with_context(&["exploit"], Some(&[vec![1.0, 0.0]]), &[1.0])
                .unwrap();
        }

        // Even with high reward for "exploit", should sometimes explore
        let mut selections = HashMap::new();
        for _ in 0..100 {
            let selected = policy
                .select_with_context(&arms, Some(&[1.0, 0.0]), &mut rng)
                .unwrap()
                .unwrap();
            *selections.entry(selected).or_insert(0) += 1;
        }

        // Should have selected "explore" at least once due to UCB
        assert!(selections.contains_key(&"explore"));
    }
}
