use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;
use rand::RngCore;

use crate::neighborhood::distance::{DistanceMetric, Euclidean};
use crate::policy::Policy;

/// K-Nearest Neighbors contextual policy wrapper
///
/// This policy uses k-nearest neighbors to select historical observations
/// that are most similar to the current context, then trains an underlying
/// policy on those observations to make predictions.
///
/// # Type Parameters
/// - `A`: The arm type
/// - `P`: The underlying policy type that will be trained on neighbor data
/// - `D`: The distance metric type
///
/// # Example
/// ```
/// use trashpanda::{Bandit, contextual::lingreedy::LinGreedy, neighborhood::knearest::KNearest};
/// use trashpanda::neighborhood::distance::Euclidean;
///
/// let underlying = LinGreedy::new(0.1, 1.0, 2); // contextual policy
/// let policy = KNearest::new(underlying, 3, Euclidean);
/// let mut bandit = Bandit::new(vec!["arm1", "arm2"], policy).unwrap();
/// ```
#[derive(Clone)]
pub struct KNearest<A, P, D = Euclidean> {
    /// The underlying policy to train on neighbor data
    underlying_policy: P,
    /// Number of nearest neighbors to use
    k: usize,
    /// Distance metric
    metric: D,
    /// Historical decisions
    decisions: Vec<A>,
    /// Historical rewards
    rewards: Vec<f64>,
    /// Historical contexts
    contexts: Vec<Vec<f64>>,
}

impl<A, P, D> KNearest<A, P, D>
where
    A: Clone + Eq + Hash,
    P: for<'a> Policy<A, &'a [f64]> + Clone,
    D: DistanceMetric,
{
    /// Create a new KNearest policy
    ///
    /// # Arguments
    /// - `underlying_policy`: The policy to use for predictions based on neighbor data
    /// - `k`: Number of nearest neighbors to consider
    /// - `metric`: Distance metric to use
    #[must_use]
    pub fn new(underlying_policy: P, k: usize, metric: D) -> Self {
        assert!(k > 0, "k must be greater than 0");
        Self {
            underlying_policy,
            k,
            metric,
            decisions: Vec::new(),
            rewards: Vec::new(),
            contexts: Vec::new(),
        }
    }

    /// Create a new KNearest policy with default Euclidean distance
    #[must_use]
    pub fn euclidean(underlying_policy: P, k: usize) -> KNearest<A, P, Euclidean>
    where
        Euclidean: DistanceMetric,
    {
        KNearest::new(underlying_policy, k, Euclidean)
    }

    /// Find k nearest neighbors to the given context
    fn find_neighbors(&self, context: &[f64]) -> Vec<usize> {
        if self.contexts.is_empty() {
            return Vec::new();
        }

        // Calculate distances to all historical contexts
        let mut distances: Vec<(usize, f64)> = self
            .contexts
            .iter()
            .enumerate()
            .map(|(idx, hist_context)| (idx, self.metric.distance(hist_context, context)))
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k = self.k.min(distances.len());
        distances.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
}

impl<A, P, D> Policy<A, &[f64]> for KNearest<A, P, D>
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

        // Find k nearest neighbors
        let neighbor_indices = self.find_neighbors(context);

        if neighbor_indices.is_empty() {
            // No neighbors found (shouldn't happen), select randomly
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

        // Find k nearest neighbors
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
