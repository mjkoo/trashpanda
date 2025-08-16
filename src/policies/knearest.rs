use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;
use rand::RngCore;

use super::Policy;

/// K-Nearest Neighbors contextual policy wrapper
///
/// This policy uses k-nearest neighbors to select historical observations
/// that are most similar to the current context, then trains an underlying
/// policy on those observations to make predictions.
///
/// # Type Parameters
/// - `A`: The arm type
/// - `P`: The underlying policy type that will be trained on neighbor data
///
/// # Example
/// ```
/// use trashpanda::Bandit;
/// use trashpanda::policies::{KNearest, LinGreedy};
///
/// let underlying = LinGreedy::new(0.1, 1.0, 2); // contextual policy
/// let policy = KNearest::new(underlying, 3, "euclidean");
/// let mut bandit = Bandit::new(vec!["arm1", "arm2"], policy).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct KNearest<A, P> {
    /// The underlying policy to train on neighbor data
    underlying_policy: P,
    /// Number of nearest neighbors to use
    k: usize,
    /// Distance metric (e.g., "euclidean", "manhattan", "cosine")
    metric: String,
    /// Historical decisions
    decisions: Vec<A>,
    /// Historical rewards
    rewards: Vec<f64>,
    /// Historical contexts
    contexts: Vec<Vec<f64>>,
}

impl<A, P> KNearest<A, P>
where
    A: Clone + Eq + Hash,
    P: for<'a> Policy<A, &'a [f64]> + Clone,
{
    /// Create a new KNearest policy
    ///
    /// # Arguments
    /// - `underlying_policy`: The policy to use for predictions based on neighbor data
    /// - `k`: Number of nearest neighbors to consider
    /// - `metric`: Distance metric to use ("euclidean", "manhattan", "cosine", etc.)
    #[must_use]
    pub fn new(underlying_policy: P, k: usize, metric: impl Into<String>) -> Self {
        assert!(k > 0, "k must be greater than 0");
        Self {
            underlying_policy,
            k,
            metric: metric.into(),
            decisions: Vec::new(),
            rewards: Vec::new(),
            contexts: Vec::new(),
        }
    }

    /// Calculate distance between two context vectors
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");

        match self.metric.as_str() {
            "euclidean" => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            "manhattan" => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
            "cosine" => {
                let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }
            _ => panic!("Unsupported metric: {}", self.metric),
        }
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
            .map(|(idx, hist_context)| (idx, self.distance(hist_context, context)))
            .collect();

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k = self.k.min(distances.len());
        distances.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
}

impl<A, P> Policy<A, &[f64]> for KNearest<A, P>
where
    A: Clone + Eq + Hash,
    P: for<'a> Policy<A, &'a [f64]> + Clone,
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
