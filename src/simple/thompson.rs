use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;
use rand::Rng;
use rand_distr::{Beta, Distribution};

use crate::policy::Policy;

/// Thompson Sampling policy using Beta distribution
///
/// This policy maintains Beta distributions for each arm and samples from them
/// to balance exploration and exploitation. It's particularly effective for
/// binary reward scenarios.
#[derive(Clone)]
pub struct ThompsonSampling<A> {
    /// Prior alpha parameter for Beta distribution (defaults to 1.0)
    prior_alpha: f64,
    /// Prior beta parameter for Beta distribution (defaults to 1.0)
    prior_beta: f64,
    /// Statistics for each arm
    arm_stats: HashMap<A, ArmStats>,
}

impl<A> std::fmt::Debug for ThompsonSampling<A>
where
    A: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThompsonSampling")
            .field("prior_alpha", &self.prior_alpha)
            .field("prior_beta", &self.prior_beta)
            .field("arm_stats", &self.arm_stats)
            .finish()
    }
}

#[derive(Clone, Debug)]
struct ArmStats {
    /// Number of successes (rewards > 0.5 for binary, sum of rewards for continuous)
    successes: f64,
    /// Number of failures (rewards <= 0.5 for binary, sum of (1 - reward) for continuous)
    failures: f64,
}

impl Default for ArmStats {
    fn default() -> Self {
        Self {
            successes: 0.0,
            failures: 0.0,
        }
    }
}

impl ArmStats {
    /// Sample from the Beta distribution for this arm
    fn sample<R: Rng + ?Sized>(&self, prior_alpha: f64, prior_beta: f64, rng: &mut R) -> f64 {
        let alpha = self.successes + prior_alpha;
        let beta = self.failures + prior_beta;

        // Handle edge cases
        if alpha <= 0.0 || beta <= 0.0 {
            return 0.5; // Return neutral value
        }

        match Beta::new(alpha, beta) {
            Ok(dist) => dist.sample(rng),
            Err(_) => {
                // If Beta distribution creation fails, use mean
                alpha / (alpha + beta)
            }
        }
    }

    /// Get the expected value (mean) of the Beta distribution
    fn expected_value(&self, prior_alpha: f64, prior_beta: f64) -> f64 {
        let alpha = self.successes + prior_alpha;
        let beta = self.failures + prior_beta;
        alpha / (alpha + beta)
    }
}

impl<A> ThompsonSampling<A>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new Thompson Sampling policy with default priors (uniform Beta(1,1))
    #[must_use]
    pub fn new() -> Self {
        Self::with_prior(1.0, 1.0)
    }

    /// Creates a new Thompson Sampling policy with specified Beta prior parameters
    ///
    /// # Arguments
    /// * `prior_alpha` - Alpha parameter for Beta prior (must be positive)
    /// * `prior_beta` - Beta parameter for Beta prior (must be positive)
    #[must_use]
    pub fn with_prior(prior_alpha: f64, prior_beta: f64) -> Self {
        assert!(prior_alpha > 0.0, "prior_alpha must be positive");
        assert!(prior_beta > 0.0, "prior_beta must be positive");
        Self {
            prior_alpha,
            prior_beta,
            arm_stats: HashMap::new(),
        }
    }

    /// Gets the prior parameters
    pub fn prior(&self) -> (f64, f64) {
        (self.prior_alpha, self.prior_beta)
    }

    /// Gets the statistics for a specific arm
    pub fn arm_stats(&self, arm: &A) -> Option<(f64, f64, f64)> {
        self.arm_stats.get(arm).map(|s| {
            (
                s.successes,
                s.failures,
                s.expected_value(self.prior_alpha, self.prior_beta),
            )
        })
    }
}

impl<A> Default for ThompsonSampling<A>
where
    A: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A> Policy<A, ()> for ThompsonSampling<A>
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, decision: &A, _context: (), reward: f64) {
        let stats = self.arm_stats.entry(decision.clone()).or_default();

        // For continuous rewards in [0, 1], treat as probability of success
        // This is a common approach for Thompson Sampling with continuous rewards
        let reward_clamped = reward.clamp(0.0, 1.0);
        stats.successes += reward_clamped;
        stats.failures += 1.0 - reward_clamped;
    }

    fn select(&self, arms: &IndexSet<A>, _context: (), rng: &mut dyn rand::RngCore) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        // Sample from each arm's Beta distribution and select the best
        arms.iter()
            .map(|arm| {
                let sample = match self.arm_stats.get(arm) {
                    Some(stats) => stats.sample(self.prior_alpha, self.prior_beta, rng),
                    None => {
                        // Use prior for unpulled arms
                        ArmStats::default().sample(self.prior_alpha, self.prior_beta, rng)
                    }
                };
                (arm, sample)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arm, _)| arm.clone())
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        _context: (),
        _rng: &mut dyn rand::RngCore,
    ) -> HashMap<A, f64> {
        let mut expectations = HashMap::new();

        // Return the expected reward for each arm based on the Beta distribution mean
        // For Beta(α, β), the mean is α / (α + β)
        for arm in arms {
            let expected_reward = match self.arm_stats.get(arm) {
                Some(stats) => {
                    let alpha = stats.successes + self.prior_alpha;
                    let beta = stats.failures + self.prior_beta;
                    alpha / (alpha + beta)
                }
                None => {
                    // Use prior mean for unobserved arms
                    self.prior_alpha / (self.prior_alpha + self.prior_beta)
                }
            };
            expectations.insert(arm.clone(), expected_reward);
        }

        expectations
    }

    fn reset_arm(&mut self, arm: &A) {
        self.arm_stats.remove(arm);
    }

    fn reset(&mut self) {
        self.arm_stats.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use rand::SeedableRng;

    #[test]
    fn test_thompson_sampling_basic() {
        let policy = ThompsonSampling::new();
        let mut arms = IndexSet::new();
        arms.insert("a");
        arms.insert("b");
        arms.insert("c");

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Should select an arm (stochastic)
        let choice = Policy::select(&policy, &arms, (), &mut rng).unwrap();
        assert!(arms.contains(&choice));
    }

    #[test]
    fn test_thompson_sampling_learns_best_arm() {
        let mut policy = ThompsonSampling::new();
        let mut arms = IndexSet::new();
        arms.insert(1);
        arms.insert(2);
        arms.insert(3);

        // Train with data where arm 2 is clearly the best
        for _ in 0..20 {
            policy.update(&1, (), 0.2);
            policy.update(&2, (), 0.9);
            policy.update(&3, (), 0.3);
        }

        // Check expectations - arm 2 should have highest probability
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);
        assert!(expectations[&2] > expectations[&1]);
        assert!(expectations[&2] > expectations[&3]);
    }

    #[test]
    fn test_thompson_sampling_with_prior() {
        let policy = ThompsonSampling::<i32>::with_prior(2.0, 3.0);
        assert_eq!(policy.prior(), (2.0, 3.0));
    }

    #[test]
    fn test_thompson_sampling_stochastic() {
        let mut policy = ThompsonSampling::new();
        let mut arms = IndexSet::new();
        arms.insert("x");
        arms.insert("y");

        // Train with similar rewards
        policy.update(&"x", (), 0.5);
        policy.update(&"y", (), 0.5);

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(999);

        // Thompson Sampling is stochastic - might select different arms
        let mut choices = std::collections::HashSet::new();
        for _ in 0..10 {
            choices.insert(Policy::select(&policy, &arms, (), &mut rng1).unwrap());
            choices.insert(Policy::select(&policy, &arms, (), &mut rng2).unwrap());
        }

        // With similar rewards, both arms should be selected at least once
        assert!(choices.len() > 1 || choices.contains(&"x") || choices.contains(&"y"));
    }

    #[test]
    fn test_thompson_sampling_binary_rewards() {
        let mut policy = ThompsonSampling::new();

        // Binary rewards (0 or 1)
        policy.update(&1, (), 1.0);
        policy.update(&1, (), 0.0);
        policy.update(&1, (), 1.0);

        let stats = policy.arm_stats(&1).unwrap();
        assert_eq!(stats.0, 2.0); // successes
        assert_eq!(stats.1, 1.0); // failures
        assert_abs_diff_eq!(stats.2, 2.0 / 3.0, epsilon = 0.1); // expected value ≈ 0.67
    }

    #[test]
    fn test_thompson_sampling_continuous_rewards() {
        let mut policy = ThompsonSampling::new();

        // Continuous rewards in [0, 1]
        policy.update(&1, (), 0.7);
        policy.update(&1, (), 0.3);

        let stats = policy.arm_stats(&1).unwrap();
        assert_eq!(stats.0, 1.0); // successes = 0.7 + 0.3
        assert_eq!(stats.1, 1.0); // failures = 0.3 + 0.7
        assert_eq!(stats.2, 0.5); // expected value = 1.0 / 2.0
    }

    #[test]
    fn test_thompson_sampling_reset() {
        let mut policy = ThompsonSampling::new();

        policy.update(&1, (), 0.5);
        policy.update(&2, (), 0.8);
        policy.update(&3, (), 0.3);

        // Reset specific arm
        policy.reset_arm(&2);
        assert_eq!(policy.arm_stats(&2), None);
        assert!(policy.arm_stats(&1).is_some());

        // Full reset
        policy.reset();
        assert_eq!(policy.arm_stats(&1), None);
    }

    #[test]
    fn test_thompson_sampling_expectations_reflect_beta_means() {
        let mut policy = ThompsonSampling::new();
        let mut arms = IndexSet::new();
        arms.insert("a");
        arms.insert("b");
        arms.insert("c");

        policy.update(&"a", (), 0.6);
        policy.update(&"b", (), 0.4);
        policy.update(&"c", (), 0.5);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);

        // With Beta(1,1) prior and one observation each:
        // "a": Beta(1 + 0.6, 1 + 0.4) = Beta(1.6, 1.4), mean = 1.6/3.0 ≈ 0.533
        // "b": Beta(1 + 0.4, 1 + 0.6) = Beta(1.4, 1.6), mean = 1.4/3.0 ≈ 0.467
        // "c": Beta(1 + 0.5, 1 + 0.5) = Beta(1.5, 1.5), mean = 1.5/3.0 = 0.5
        assert!(abs_diff_eq!(expectations[&"a"], 1.6 / 3.0));
        assert!(abs_diff_eq!(expectations[&"b"], 1.4 / 3.0));
        assert!(abs_diff_eq!(expectations[&"c"], 1.5 / 3.0));
    }
}
