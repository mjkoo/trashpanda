use crate::policies::Policy;
use crate::regression::RidgeRegression;
use approx::abs_diff_eq;
use indexmap::IndexSet;
use rand::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

/// Linear Epsilon-Greedy (LinGreedy) policy for contextual bandits
///
/// LinGreedy uses ridge regression to model the expected reward for each arm
/// given context features, and uses epsilon-greedy exploration.
///
/// The algorithm maintains a separate ridge regression model for each arm
/// and selects the best arm with probability (1-epsilon) or a random arm
/// with probability epsilon.
#[derive(Debug, Clone)]
pub struct LinGreedy<A> {
    /// Exploration probability
    epsilon: f64,
    /// Regularization parameter for ridge regression
    l2_lambda: f64,
    /// Number of context features
    num_features: usize,
    /// Ridge regression model for each arm
    arm_models: HashMap<A, RidgeRegression>,
}

impl<A> LinGreedy<A>
where
    A: Clone + Eq + Hash,
{
    /// Create a new Linear Epsilon-Greedy policy
    ///
    /// # Arguments
    /// * `epsilon` - Exploration probability (typically between 0.01 and 0.3)
    /// * `l2_lambda` - L2 regularization parameter (typically 1.0)
    /// * `num_features` - Number of context features
    #[must_use]
    pub fn new(epsilon: f64, l2_lambda: f64, num_features: usize) -> Self {
        Self {
            epsilon,
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

    /// Get the predicted reward for an arm given context
    fn get_prediction(&self, arm: &A, context: &[f64]) -> f64 {
        if let Some(model) = self.arm_models.get(arm) {
            model.predict(context)
        } else {
            // Uninitialized arm gets zero prediction
            0.0
        }
    }

    /// Get epsilon value (could be made dynamic in the future)
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Set epsilon value for decaying exploration
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon.clamp(0.0, 1.0);
    }
}

impl<A> Policy<A, &[f64]> for LinGreedy<A>
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, decision: &A, context: &[f64], reward: f64) {
        let model = self.get_or_create_model(decision);
        model.fit(context, reward);
    }

    fn select(
        &self,
        arms: &IndexSet<A>,
        context: &[f64],
        rng: &mut dyn rand::RngCore,
    ) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        // Epsilon-greedy selection
        if rng.random::<f64>() < self.epsilon {
            // Explore: select random arm
            let idx = rng.random_range(0..arms.len());
            arms.get_index(idx).cloned()
        } else {
            // Exploit: select best arm based on predictions
            let mut best_arms = Vec::new();
            let mut best_reward = f64::NEG_INFINITY;

            for arm in arms {
                let predicted_reward = self.get_prediction(arm, context);

                if abs_diff_eq!(predicted_reward, best_reward) {
                    // Tie - add to best arms
                    best_arms.push(arm.clone());
                } else if predicted_reward > best_reward {
                    // New best
                    best_reward = predicted_reward;
                    best_arms.clear();
                    best_arms.push(arm.clone());
                }
            }

            // Break ties randomly
            best_arms.choose(rng).cloned()
        }
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        context: &[f64],
        rng: &mut dyn rand::RngCore,
    ) -> HashMap<A, f64> {
        let mut expectations = HashMap::new();

        // MABWiser's LinGreedy implements epsilon-greedy mixing in predict_expectations:
        // With probability epsilon: return random values [0,1]
        // With probability (1-epsilon): return ridge regression predictions
        if rng.random::<f64>() < self.epsilon {
            // Exploration: return random expectations
            for arm in arms {
                expectations.insert(arm.clone(), rng.random::<f64>());
            }
        } else {
            // Exploitation: return actual ridge regression predictions
            for arm in arms {
                expectations.insert(arm.clone(), self.get_prediction(arm, context));
            }
        }

        expectations
    }

    fn reset_arm(&mut self, arm: &A) {
        if let Some(model) = self.arm_models.get_mut(arm) {
            model.reset();
        }
    }

    fn reset(&mut self) {
        self.arm_models.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_lingreedy_creation() {
        let policy: LinGreedy<&str> = LinGreedy::new(0.1, 1.0, 3);
        assert_eq!(policy.num_features, 3);
        assert_eq!(policy.epsilon(), 0.1);
    }

    #[test]
    fn test_lingreedy_epsilon_adjustment() {
        let mut policy: LinGreedy<i32> = LinGreedy::new(0.5, 1.0, 2);
        assert_eq!(policy.epsilon(), 0.5);

        policy.set_epsilon(0.1);
        assert_eq!(policy.epsilon(), 0.1);

        // Test clamping
        policy.set_epsilon(1.5);
        assert_eq!(policy.epsilon(), 1.0);

        policy.set_epsilon(-0.1);
        assert_eq!(policy.epsilon(), 0.0);
    }

    #[test]
    fn test_lingreedy_select() {
        let policy: LinGreedy<i32> = LinGreedy::new(0.0, 1.0, 2); // No exploration
        let arms = IndexSet::from([1, 2, 3]);
        let context = vec![0.5, 0.5];
        let mut rng = StdRng::seed_from_u64(42);

        // Should select from arms
        let selected = policy.select(&arms, &context, &mut rng);
        assert!(selected.is_some());
        assert!(arms.contains(&selected.unwrap()));
    }

    #[test]
    fn test_lingreedy_update_and_predict() {
        let mut policy: LinGreedy<&str> = LinGreedy::new(0.0, 1.0, 2); // No exploration
        let arms = IndexSet::from(["a", "b"]);

        // Train with some data
        policy.update(&"a", &[1.0, 0.0], 1.0);
        policy.update(&"a", &[0.8, 0.2], 0.8);
        policy.update(&"b", &[0.0, 1.0], 0.2);
        policy.update(&"b", &[0.2, 0.8], 0.3);

        // Check expectations
        let context = vec![1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, &context, &mut rng);

        // Arm "a" should have higher expectation for context [1.0, 0.0]
        assert!(expectations[&"a"] > expectations[&"b"]);
    }

    #[test]
    fn test_lingreedy_reset() {
        let mut policy: LinGreedy<i32> = LinGreedy::new(0.1, 1.0, 2);

        // Train with some data
        policy.update(&1, &[1.0, 0.0], 1.0);

        // Reset and check that model is cleared
        <LinGreedy<i32> as Policy<i32, &[f64]>>::reset(&mut policy);

        let arms = IndexSet::from([1, 2]);
        let context = vec![1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, &context, &mut rng);

        // After reset, all arms should have zero expectation
        assert_eq!(expectations[&1], 0.0);
        assert_eq!(expectations[&2], 0.0);
    }

    #[test]
    fn test_lingreedy_exploration() {
        let mut policy: LinGreedy<&str> = LinGreedy::new(0.2, 1.0, 2); // 20% exploration
        let arms = IndexSet::from(["exploit", "explore"]);
        let mut rng = StdRng::seed_from_u64(42);

        // Train "exploit" arm to have high reward
        for _ in 0..20 {
            policy.update(&"exploit", &[1.0, 0.0], 1.0);
        }
        // Train "explore" arm to have low reward
        for _ in 0..20 {
            policy.update(&"explore", &[1.0, 0.0], 0.0);
        }

        // Track selections over many iterations
        let mut selections = HashMap::new();
        let context = vec![1.0, 0.0];
        for _ in 0..1000 {
            let selected = policy.select(&arms, &context, &mut rng).unwrap();
            *selections.entry(selected).or_insert(0) += 1;
        }

        // With epsilon=0.2, "explore" should be selected around 10% of the time
        // (20% random selection * 50% chance of selecting explore when random)
        // However, when exploiting (80%), it will always select "exploit"
        // So expected is roughly 100 times out of 1000
        let explore_count = *selections.get(&"explore").unwrap_or(&0);
        assert!(
            explore_count > 50,
            "Expected at least 50 explorations, got {}",
            explore_count
        );
        assert!(
            explore_count < 200,
            "Expected at most 200 explorations, got {}",
            explore_count
        );
    }

    #[test]
    fn test_lingreedy_pure_exploitation() {
        let mut policy: LinGreedy<i32> = LinGreedy::new(0.0, 1.0, 2); // No exploration
        let arms = IndexSet::from([1, 2]);
        let mut rng = StdRng::seed_from_u64(42);

        // Train arm 1 to be clearly better
        policy.update(&1, &[1.0, 0.0], 1.0);
        policy.update(&1, &[0.9, 0.1], 0.9);
        policy.update(&2, &[1.0, 0.0], 0.1);
        policy.update(&2, &[0.9, 0.1], 0.2);

        // With epsilon=0, should always select arm 1
        let context = vec![1.0, 0.0];
        for _ in 0..100 {
            let selected = policy.select(&arms, &context, &mut rng).unwrap();
            assert_eq!(selected, 1);
        }
    }
}
