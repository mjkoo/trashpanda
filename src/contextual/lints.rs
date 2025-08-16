use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

use crate::policy::Policy;
use crate::regression::RidgeRegression;

/// Linear Thompson Sampling (LinTS) policy for contextual bandits
///
/// LinTS uses ridge regression to model the expected reward for each arm
/// given context features, and samples from the posterior distribution
/// for exploration.
///
/// The algorithm maintains a separate ridge regression model for each arm
/// and samples regression coefficients from their posterior distribution.
#[derive(Debug, Clone)]
pub struct LinTs<A> {
    /// Variance parameter for posterior sampling (controls exploration)
    v: f64,
    /// Regularization parameter for ridge regression
    l2_lambda: f64,
    /// Number of context features
    num_features: usize,
    /// Ridge regression model for each arm
    arm_models: HashMap<A, RidgeRegression>,
}

impl<A> LinTs<A>
where
    A: Clone + Eq + Hash,
{
    /// Create a new Linear Thompson Sampling policy
    ///
    /// # Arguments
    /// * `v` - Variance parameter for posterior sampling (typically 1.0)
    /// * `l2_lambda` - L2 regularization parameter (typically 1.0)
    /// * `num_features` - Number of context features
    #[must_use]
    pub fn new(v: f64, l2_lambda: f64, num_features: usize) -> Self {
        Self {
            v,
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

    /// Sample from the posterior distribution for an arm
    fn sample_posterior(&self, arm: &A, context: &[f64], rng: &mut dyn RngCore) -> f64 {
        if let Some(model) = self.arm_models.get(arm) {
            // Sample beta from multivariate normal: N(beta_hat, v^2 * A_inv)
            let x_vec = Array1::from(context.to_vec());

            // Sample perturbed beta coefficients
            let normal = Normal::new(0.0, 1.0).unwrap();
            let mut beta_sample = model.beta.clone();

            // Add noise to each coefficient: beta_sample = beta + sqrt(v) * A_inv^(1/2) * z
            // For simplicity, we use diagonal approximation
            for i in 0..self.num_features {
                let noise = normal.sample(rng);
                let variance = self.v * model.a_inv[[i, i]].sqrt();
                beta_sample[i] += noise * variance;
            }

            // Return x^T * beta_sample
            x_vec.dot(&beta_sample)
        } else {
            // Uninitialized arm - sample from prior with high variance
            let normal = Normal::new(0.0, self.v).unwrap();
            normal.sample(rng)
        }
    }
}

impl<A> Policy<A, &[f64]> for LinTs<A>
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

        // Sample from posterior for each arm
        let mut best_arm = None;
        let mut best_sample = f64::NEG_INFINITY;

        for arm in arms {
            let sample = self.sample_posterior(arm, context, rng);

            if sample > best_sample {
                best_sample = sample;
                best_arm = Some(arm.clone());
            }
        }

        best_arm
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        context: &[f64],
        _rng: &mut dyn rand::RngCore,
    ) -> HashMap<A, f64> {
        let mut expectations = HashMap::new();

        for arm in arms {
            if let Some(model) = self.arm_models.get(arm) {
                // For LinTS, expectations should return the mean prediction (not sampled values)
                // This matches MABWiser's behavior where the base ridge regression returns mean
                expectations.insert(arm.clone(), model.predict(context));
            } else {
                // Uninitialized arms get zero expectation
                expectations.insert(arm.clone(), 0.0);
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
    fn test_lints_creation() {
        let policy: LinTs<&str> = LinTs::new(1.0, 1.0, 3);
        assert_eq!(policy.num_features, 3);
    }

    #[test]
    fn test_lints_select_explores() {
        let policy: LinTs<i32> = LinTs::new(1.0, 1.0, 2);
        let arms = IndexSet::from([1, 2, 3]);
        let context = vec![0.5, 0.5];
        let mut rng = StdRng::seed_from_u64(42);

        // Should select from arms
        let selected = policy.select(&arms, &context, &mut rng);
        assert!(selected.is_some());
        assert!(arms.contains(&selected.unwrap()));
    }

    #[test]
    fn test_lints_update_and_predict() {
        let mut policy: LinTs<&str> = LinTs::new(0.5, 1.0, 2);
        let arms = IndexSet::from(["a", "b"]);

        // Train with some data
        policy.update(&"a", &[1.0, 0.0], 1.0);
        policy.update(&"a", &[0.8, 0.2], 0.8);
        policy.update(&"b", &[0.0, 1.0], 0.2);

        // Check expectations
        let context = vec![1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, &context, &mut rng);

        // Arm "a" should have higher expectation for context [1.0, 0.0]
        assert!(expectations[&"a"] > expectations[&"b"]);
    }

    #[test]
    fn test_lints_reset() {
        let mut policy: LinTs<i32> = LinTs::new(1.0, 1.0, 2);

        // Train with some data
        policy.update(&1, &[1.0, 0.0], 1.0);

        // Reset and check that model is cleared
        <LinTs<i32> as Policy<i32, &[f64]>>::reset(&mut policy);

        let arms = IndexSet::from([1, 2]);
        let context = vec![1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, &context, &mut rng);

        // After reset, all arms should have zero expectation
        assert_eq!(expectations[&1], 0.0);
        assert_eq!(expectations[&2], 0.0);
    }

    #[test]
    fn test_lints_exploration() {
        let mut policy: LinTs<&str> = LinTs::new(2.0, 1.0, 2); // Higher v for more exploration
        let arms = IndexSet::from(["exploit", "explore"]);
        let mut rng = StdRng::seed_from_u64(42);

        // Train "exploit" arm to have high reward
        for _ in 0..10 {
            policy.update(&"exploit", &[1.0, 0.0], 1.0);
        }

        // Even with high reward for "exploit", should sometimes explore
        let mut selections = HashMap::new();
        let context = vec![1.0, 0.0];
        for _ in 0..100 {
            let selected = policy.select(&arms, &context, &mut rng).unwrap();
            *selections.entry(selected).or_insert(0) += 1;
        }

        // Should have selected "explore" at least once due to Thompson sampling
        assert!(selections.contains_key(&"explore"));
    }
}
