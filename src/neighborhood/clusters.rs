use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;
use rand::RngCore;

use crate::neighborhood::distance::{DistanceMetric, Euclidean};
use crate::policy::Policy;

/// K-means clustering contextual policy wrapper
///
/// This policy partitions the context space into k clusters using k-means,
/// then trains separate underlying policies for each cluster.
///
/// # Type Parameters
/// - `A`: The arm type
/// - `P`: The underlying policy type that will be trained on cluster data
/// - `D`: The distance metric type
///
/// # Example
/// ```
/// use trashpanda::{Bandit, contextual::lingreedy::LinGreedy, neighborhood::clusters::Clusters};
/// use trashpanda::neighborhood::distance::Euclidean;
///
/// let underlying = LinGreedy::new(0.1, 1.0, 2); // contextual policy
/// let policy = Clusters::new(underlying, 3, Euclidean);
/// let mut bandit = Bandit::new(vec!["arm1", "arm2"], policy).unwrap();
/// ```
#[derive(Clone)]
pub struct Clusters<A, P, D = Euclidean> {
    /// The underlying policy template (cloned for each cluster)
    _underlying_policy: P,
    /// Number of clusters
    n_clusters: usize,
    /// Distance metric
    metric: D,
    /// Maximum iterations for k-means
    max_iter: usize,
    /// Cluster centroids
    centroids: Vec<Vec<f64>>,
    /// Policies for each cluster
    cluster_policies: Vec<P>,
    /// Historical decisions
    decisions: Vec<A>,
    /// Historical rewards
    rewards: Vec<f64>,
    /// Historical contexts
    contexts: Vec<Vec<f64>>,
    /// Cluster assignments for historical data
    cluster_assignments: Vec<usize>,
}

impl<A, P, D> Clusters<A, P, D>
where
    A: Clone + Eq + Hash,
    P: for<'a> Policy<A, &'a [f64]> + Clone,
    D: DistanceMetric,
{
    /// Create a new Clusters policy
    ///
    /// # Arguments
    /// - `underlying_policy`: The policy template to use for each cluster
    /// - `n_clusters`: Number of clusters to create
    /// - `metric`: Distance metric to use
    #[must_use]
    pub fn new(underlying_policy: P, n_clusters: usize, metric: D) -> Self {
        assert!(n_clusters > 0, "n_clusters must be greater than 0");

        // Initialize cluster policies
        let cluster_policies = (0..n_clusters).map(|_| underlying_policy.clone()).collect();

        Self {
            _underlying_policy: underlying_policy,
            n_clusters,
            metric,
            max_iter: 100,
            centroids: Vec::new(),
            cluster_policies,
            decisions: Vec::new(),
            rewards: Vec::new(),
            contexts: Vec::new(),
            cluster_assignments: Vec::new(),
        }
    }

    /// Create a new Clusters policy with default Euclidean distance
    #[must_use]
    pub fn euclidean(underlying_policy: P, n_clusters: usize) -> Clusters<A, P, Euclidean>
    where
        Euclidean: DistanceMetric,
    {
        Clusters::new(underlying_policy, n_clusters, Euclidean)
    }

    /// Set maximum iterations for k-means
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Find the nearest cluster centroid for a given context
    fn find_nearest_cluster(&self, context: &[f64]) -> Option<usize> {
        if self.centroids.is_empty() {
            return None;
        }

        self.centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| (idx, self.metric.distance(centroid, context)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| idx)
    }

    /// Run k-means clustering on the historical contexts
    fn update_clusters(&mut self) {
        if self.contexts.len() < self.n_clusters {
            // Not enough data for clustering
            return;
        }

        let n_features = self.contexts[0].len();

        // Initialize centroids using k-means++ algorithm
        self.centroids = self.kmeans_plus_plus_init(n_features);

        // Run k-means iterations
        for _ in 0..self.max_iter {
            // Assign points to nearest centroids
            let new_assignments: Vec<usize> = self
                .contexts
                .iter()
                .map(|context| self.find_nearest_cluster(context).unwrap())
                .collect();

            // Check for convergence
            if new_assignments == self.cluster_assignments {
                break;
            }
            self.cluster_assignments = new_assignments;

            // Update centroids
            let mut new_centroids = vec![vec![0.0; n_features]; self.n_clusters];
            let mut counts = vec![0usize; self.n_clusters];

            for (context, &cluster) in self.contexts.iter().zip(&self.cluster_assignments) {
                for (j, &val) in context.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
                counts[cluster] += 1;
            }

            // Average to get new centroids
            for (centroid, count) in new_centroids.iter_mut().zip(&counts) {
                if *count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= *count as f64;
                    }
                }
            }

            self.centroids = new_centroids;
        }

        // Retrain cluster policies with assigned data
        self.retrain_cluster_policies();
    }

    /// Initialize centroids using k-means++ algorithm
    fn kmeans_plus_plus_init(&self, n_features: usize) -> Vec<Vec<f64>> {
        let mut centroids = Vec::new();

        // Choose first centroid randomly from data points
        if !self.contexts.is_empty() {
            centroids.push(self.contexts[0].clone());
        }

        // Choose remaining centroids with probability proportional to squared distance
        while centroids.len() < self.n_clusters && centroids.len() < self.contexts.len() {
            let mut max_dist = 0.0;
            let mut best_idx = 0;

            for (idx, context) in self.contexts.iter().enumerate() {
                let min_dist = centroids
                    .iter()
                    .map(|centroid| self.metric.distance(context, centroid))
                    .fold(f64::MAX, f64::min);

                if min_dist > max_dist {
                    max_dist = min_dist;
                    best_idx = idx;
                }
            }

            centroids.push(self.contexts[best_idx].clone());
        }

        // If not enough data points, fill with zeros
        while centroids.len() < self.n_clusters {
            centroids.push(vec![0.0; n_features]);
        }

        centroids
    }

    /// Retrain cluster policies based on current assignments
    fn retrain_cluster_policies(&mut self) {
        // Reset all cluster policies
        for policy in &mut self.cluster_policies {
            policy.reset();
        }

        // Train each policy with its assigned data
        for (&cluster, decision, reward, context) in self
            .cluster_assignments
            .iter()
            .zip(&self.decisions)
            .zip(&self.rewards)
            .zip(&self.contexts)
            .map(|(((a, b), c), d)| (a, b, c, d))
        {
            self.cluster_policies[cluster].update(decision, context, *reward);
        }
    }
}

impl<A, P, D> Policy<A, &[f64]> for Clusters<A, P, D>
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

        // Assign to nearest cluster if clusters exist
        if let Some(cluster) = self.find_nearest_cluster(context) {
            self.cluster_assignments.push(cluster);
            self.cluster_policies[cluster].update(decision, context, reward);
        } else {
            // No clusters yet, assign to cluster 0
            self.cluster_assignments.push(0);
            if !self.cluster_policies.is_empty() {
                self.cluster_policies[0].update(decision, context, reward);
            }
        }

        // Periodically update clusters (e.g., every 10 samples)
        if self.contexts.len() % 10 == 0 && self.contexts.len() >= self.n_clusters {
            self.update_clusters();
        }
    }

    fn select(&self, arms: &IndexSet<A>, context: &[f64], rng: &mut dyn RngCore) -> Option<A> {
        if self.contexts.is_empty() || self.centroids.is_empty() {
            // No historical data or clusters, select randomly
            if arms.is_empty() {
                return None;
            }
            let idx = rand::seq::index::sample(rng, arms.len(), 1).index(0);
            return arms.get_index(idx).cloned();
        }

        // Find nearest cluster
        if let Some(cluster) = self.find_nearest_cluster(context) {
            // Use the corresponding cluster policy
            self.cluster_policies[cluster].select(arms, context, rng)
        } else {
            // Shouldn't happen, but fallback to random
            if arms.is_empty() {
                None
            } else {
                let idx = rand::seq::index::sample(rng, arms.len(), 1).index(0);
                arms.get_index(idx).cloned()
            }
        }
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        context: &[f64],
        rng: &mut dyn RngCore,
    ) -> HashMap<A, f64> {
        if self.contexts.is_empty() || self.centroids.is_empty() {
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

        // Find nearest cluster
        if let Some(cluster) = self.find_nearest_cluster(context) {
            // Use the corresponding cluster policy
            self.cluster_policies[cluster].expectations(arms, context, rng)
        } else {
            // Shouldn't happen, but return uniform
            let uniform_value = if arms.is_empty() {
                0.0
            } else {
                1.0 / arms.len() as f64
            };
            arms.iter()
                .map(|arm| (arm.clone(), uniform_value))
                .collect()
        }
    }

    fn reset_arm(&mut self, arm: &A) {
        // Remove historical data for this arm
        let mut i = 0;
        while i < self.decisions.len() {
            if &self.decisions[i] == arm {
                self.decisions.remove(i);
                self.rewards.remove(i);
                self.contexts.remove(i);
                self.cluster_assignments.remove(i);
            } else {
                i += 1;
            }
        }

        // Reset in all cluster policies
        for policy in &mut self.cluster_policies {
            policy.reset_arm(arm);
        }

        // Update clusters if we have enough data
        if self.contexts.len() >= self.n_clusters {
            self.update_clusters();
        }
    }

    fn reset(&mut self) {
        self.decisions.clear();
        self.rewards.clear();
        self.contexts.clear();
        self.cluster_assignments.clear();
        self.centroids.clear();

        // Reset all cluster policies
        for policy in &mut self.cluster_policies {
            policy.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual::lingreedy::LinGreedy;
    use crate::neighborhood::distance::{Euclidean, Manhattan};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_clusters_policy() {
        let underlying: LinGreedy<&str> = LinGreedy::new(0.1, 1.0, 2);
        let mut policy = Clusters::new(underlying, 2, Euclidean);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let arms: IndexSet<&str> = ["A", "B", "C"].into_iter().collect();

        // Train with data that should form two clusters
        // Cluster 1: around [0, 0]
        policy.update(&"A", &[0.0, 0.0], 1.0);
        policy.update(&"A", &[0.1, 0.1], 0.9);
        policy.update(&"B", &[0.2, 0.0], 0.8);

        // Cluster 2: around [5, 5]
        policy.update(&"C", &[5.0, 5.0], 0.7);
        policy.update(&"C", &[5.1, 4.9], 0.6);
        policy.update(&"B", &[4.9, 5.1], 0.5);

        // Force cluster update
        for _ in 0..4 {
            policy.update(&"A", &[0.0, 0.0], 1.0);
        }

        // Predict with a context in cluster 1
        let context1 = vec![0.15, 0.15];
        let selection1 = policy.select(&arms, &context1, &mut rng);
        assert!(selection1.is_some());

        // Predict with a context in cluster 2
        let context2 = vec![5.05, 5.05];
        let selection2 = policy.select(&arms, &context2, &mut rng);
        assert!(selection2.is_some());
    }

    #[test]
    fn test_clusters_initialization() {
        let underlying: LinGreedy<&str> = LinGreedy::new(0.1, 1.0, 2);
        let policy = Clusters::new(underlying, 3, Manhattan).with_max_iter(50);

        assert_eq!(policy.n_clusters, 3);
        assert_eq!(policy.max_iter, 50);
        assert_eq!(policy.cluster_policies.len(), 3);
    }
}
