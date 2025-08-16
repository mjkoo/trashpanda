use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;
use rand::RngCore;

use crate::neighborhood::distance::{DistanceMetric, Euclidean};
use crate::policy::Policy;

/// Radius-based neighborhood contextual policy wrapper
///
/// This policy selects historical observations within a fixed radius
/// of the current context, then trains an underlying policy on those
/// observations to make predictions.
///
/// # Type Parameters
/// - `A`: The arm type
/// - `P`: The underlying policy type that will be trained on neighbor data
/// - `D`: The distance metric type
///
/// # Example
/// ```
/// use trashpanda::{Bandit, contextual::lingreedy::LinGreedy, neighborhood::radius::Radius};
/// use trashpanda::neighborhood::distance::Euclidean;
///
/// let underlying = LinGreedy::new(0.1, 1.0, 2); // contextual policy
/// let policy = Radius::new(underlying, 0.5, Euclidean);
/// let mut bandit = Bandit::new(vec!["arm1", "arm2"], policy).unwrap();
/// ```
#[derive(Clone)]
pub struct Radius<A, P, D = Euclidean> {
    /// The underlying policy to train on neighbor data
    underlying_policy: P,
    /// Maximum distance radius for neighbors
    radius: f64,
    /// Distance metric
    metric: D,
    /// Minimum number of neighbors required (fallback to all if fewer)
    min_neighbors: usize,
    /// Historical decisions
    decisions: Vec<A>,
    /// Historical rewards
    rewards: Vec<f64>,
    /// Historical contexts
    contexts: Vec<Vec<f64>>,
}

impl<A, P, D> Radius<A, P, D>
where
    A: Clone + Eq + Hash,
    P: for<'a> Policy<A, &'a [f64]> + Clone,
    D: DistanceMetric,
{
    /// Create a new Radius policy
    ///
    /// # Arguments
    /// - `underlying_policy`: The policy to use for predictions based on neighbor data
    /// - `radius`: Maximum distance for neighbors
    /// - `metric`: Distance metric to use
    #[must_use]
    pub fn new(underlying_policy: P, radius: f64, metric: D) -> Self {
        assert!(radius > 0.0, "radius must be greater than 0");
        Self {
            underlying_policy,
            radius,
            metric,
            min_neighbors: 1,
            decisions: Vec::new(),
            rewards: Vec::new(),
            contexts: Vec::new(),
        }
    }

    /// Create a new Radius policy with default Euclidean distance
    #[must_use]
    pub fn euclidean(underlying_policy: P, radius: f64) -> Radius<A, P, Euclidean>
    where
        Euclidean: DistanceMetric,
    {
        Radius::new(underlying_policy, radius, Euclidean)
    }

    /// Set the minimum number of neighbors required
    #[must_use]
    pub fn with_min_neighbors(mut self, min_neighbors: usize) -> Self {
        self.min_neighbors = min_neighbors;
        self
    }

    /// Find neighbors within radius of the given context
    fn find_neighbors(&self, context: &[f64]) -> Vec<usize> {
        if self.contexts.is_empty() {
            return Vec::new();
        }

        // Find all contexts within the radius
        let mut neighbors: Vec<usize> = self
            .contexts
            .iter()
            .enumerate()
            .filter_map(|(idx, hist_context)| {
                let distance = self.metric.distance(hist_context, context);
                if distance <= self.radius {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        // If we have fewer neighbors than the minimum required,
        // fall back to using all historical data
        if neighbors.len() < self.min_neighbors {
            neighbors = (0..self.contexts.len()).collect();
        }

        neighbors
    }
}

impl<A, P, D> Policy<A, &[f64]> for Radius<A, P, D>
where
    A: Clone + Eq + Hash,
    P: for<'a> Policy<A, &'a [f64]> + Clone,
    D: DistanceMetric,
{
    fn update(&mut self, decision: &A, context: &[f64], reward: f64) {
        // Store the historical data
        self.decisions.push(decision.clone());
        self.rewards.push(reward);
        self.contexts.push(context.to_vec());

        // Also update the underlying policy to maintain its state
        self.underlying_policy.update(decision, context, reward);
    }

    fn select(&self, arms: &IndexSet<A>, context: &[f64], rng: &mut dyn RngCore) -> Option<A> {
        if self.contexts.is_empty() {
            // No historical data, select randomly
            if arms.is_empty() {
                return None;
            }
            let idx = rand::seq::index::sample(rng, arms.len(), 1).index(0);
            return arms.get_index(idx).cloned();
        }

        // Find neighbors within radius
        let neighbor_indices = self.find_neighbors(context);

        if neighbor_indices.is_empty() {
            // No neighbors found (shouldn't happen with fallback), select randomly
            if arms.is_empty() {
                return None;
            }
            let idx = rand::seq::index::sample(rng, arms.len(), 1).index(0);
            return arms.get_index(idx).cloned();
        }

        // Create a new policy instance and train it on neighbor data
        let mut temp_policy = self.underlying_policy.clone();

        // Reset the policy to start fresh
        temp_policy.reset();

        // Train on neighbor data
        for &idx in &neighbor_indices {
            temp_policy.update(&self.decisions[idx], &self.contexts[idx], self.rewards[idx]);
        }

        // Make prediction using the trained policy
        temp_policy.select(arms, context, rng)
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        context: &[f64],
        rng: &mut dyn RngCore,
    ) -> HashMap<A, f64> {
        if self.contexts.is_empty() {
            // No historical data, return uniform expectations
            let uniform_value = if arms.is_empty() {
                0.0
            } else {
                1.0 / arms.len() as f64
            };
            return arms
                .iter()
                .map(|arm| (arm.clone(), uniform_value))
                .collect();
        }

        // Find neighbors within radius
        let neighbor_indices = self.find_neighbors(context);

        if neighbor_indices.is_empty() {
            // No neighbors found, return uniform expectations
            let uniform_value = if arms.is_empty() {
                0.0
            } else {
                1.0 / arms.len() as f64
            };
            return arms
                .iter()
                .map(|arm| (arm.clone(), uniform_value))
                .collect();
        }

        // Create a new policy instance and train it on neighbor data
        let mut temp_policy = self.underlying_policy.clone();

        // Reset the policy to start fresh
        temp_policy.reset();

        // Train on neighbor data
        for &idx in &neighbor_indices {
            temp_policy.update(&self.decisions[idx], &self.contexts[idx], self.rewards[idx]);
        }

        // Get expectations using the trained policy
        temp_policy.expectations(arms, context, rng)
    }

    fn reset_arm(&mut self, arm: &A) {
        // Remove historical data for this arm
        let mut i = 0;
        while i < self.decisions.len() {
            if &self.decisions[i] == arm {
                self.decisions.remove(i);
                self.rewards.remove(i);
                self.contexts.remove(i);
            } else {
                i += 1;
            }
        }

        // Also reset in underlying policy
        self.underlying_policy.reset_arm(arm);
    }

    fn reset(&mut self) {
        self.decisions.clear();
        self.rewards.clear();
        self.contexts.clear();
        self.underlying_policy.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual::lingreedy::LinGreedy;
    use crate::neighborhood::distance::Euclidean;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_radius_policy() {
        let underlying: LinGreedy<&str> = LinGreedy::new(0.1, 1.0, 2);
        let mut policy = Radius::new(underlying, 1.0, Euclidean);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let arms: IndexSet<&str> = ["A", "B", "C"].into_iter().collect();

        // Train with some data
        policy.update(&"A", &[0.0, 0.0], 1.0);
        policy.update(&"B", &[1.0, 0.0], 0.5);
        policy.update(&"C", &[0.0, 1.0], 0.7);

        // Predict with a context close to [0, 0]
        let context = vec![0.1, 0.1];
        let selection = policy.select(&arms, &context, &mut rng);
        assert!(selection.is_some());

        // Predict with a context far from all training data
        let far_context = vec![10.0, 10.0];
        let selection_far = policy.select(&arms, &far_context, &mut rng);
        assert!(selection_far.is_some()); // Should fallback to all data
    }

    #[test]
    fn test_radius_with_min_neighbors() {
        let underlying: LinGreedy<&str> = LinGreedy::new(0.1, 1.0, 2);
        let mut policy = Radius::new(underlying, 0.5, Euclidean).with_min_neighbors(2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let arms: IndexSet<&str> = ["A", "B"].into_iter().collect();

        // Train with some data
        policy.update(&"A", &[0.0, 0.0], 1.0);
        policy.update(&"B", &[5.0, 5.0], 0.5);

        // Predict with a context close to only one point
        let context = vec![0.1, 0.1];
        let selection = policy.select(&arms, &context, &mut rng);
        assert!(selection.is_some()); // Should use all data due to min_neighbors
    }
}
