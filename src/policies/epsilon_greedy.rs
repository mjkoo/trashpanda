use super::Policy;
use crate::error::{PolicyError, PolicyResult};
use indexmap::IndexSet;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Epsilon-greedy policy - explores with probability epsilon, exploits otherwise
#[derive(Clone)]
pub struct EpsilonGreedy<A> {
    epsilon: f64,
    arm_stats: HashMap<A, ArmStats>,
}

impl<A> std::fmt::Debug for EpsilonGreedy<A>
where
    A: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpsilonGreedy")
            .field("epsilon", &self.epsilon)
            .field("arm_stats", &self.arm_stats)
            .finish()
    }
}

#[derive(Clone, Debug, Default)]
struct ArmStats {
    pulls: usize,
    total_reward: f64,
}

impl ArmStats {
    fn average_reward(&self) -> f64 {
        if self.pulls == 0 {
            0.0
        } else {
            self.total_reward / self.pulls as f64
        }
    }
}

impl<A> EpsilonGreedy<A>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new EpsilonGreedy policy with the given epsilon
    pub fn new(epsilon: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&epsilon),
            "epsilon must be between 0 and 1"
        );
        Self {
            epsilon,
            arm_stats: HashMap::new(),
        }
    }

    /// Gets the epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Sets the epsilon value
    pub fn set_epsilon(&mut self, epsilon: f64) {
        assert!(
            (0.0..=1.0).contains(&epsilon),
            "epsilon must be between 0 and 1"
        );
        self.epsilon = epsilon;
    }

    /// Gets the statistics for a specific arm
    pub fn arm_stats(&self, arm: &A) -> Option<(usize, f64)> {
        self.arm_stats
            .get(arm)
            .map(|s| (s.pulls, s.average_reward()))
    }
}

// Note: No more Send + Sync bounds required!
impl<A> Policy<A> for EpsilonGreedy<A>
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, decisions: &[A], rewards: &[f64]) {
        for (arm, reward) in decisions.iter().zip(rewards.iter()) {
            let stats = self.arm_stats.entry(arm.clone()).or_default();
            stats.pulls += 1;
            stats.total_reward += reward;
        }
    }

    fn select(&self, arms: &IndexSet<A>, rng: &mut dyn rand::RngCore) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        // Explore with probability epsilon
        let r: f64 = rng.random();
        if r < self.epsilon {
            // Random selection (exploration)
            let idx = rng.random_range(0..arms.len());
            arms.get_index(idx).cloned()
        } else {
            // Select arm with highest average reward (exploitation)
            self.find_best_arm(arms)
        }
    }

    fn expectations(&self, arms: &IndexSet<A>) -> HashMap<A, f64> {
        if arms.is_empty() {
            return HashMap::new();
        }

        // Find the best arm
        let best_arm = self.find_best_arm(arms);
        let best_arm_ref = best_arm.as_ref();

        // Calculate probabilities
        let explore_prob = self.epsilon / arms.len() as f64;
        let exploit_prob = 1.0 - self.epsilon;

        let mut expectations = HashMap::new();
        for arm in arms {
            let prob = if Some(arm) == best_arm_ref {
                exploit_prob + explore_prob
            } else {
                explore_prob
            };
            expectations.insert(arm.clone(), prob);
        }

        expectations
    }

    fn reset_arm(&mut self, arm: &A) {
        self.arm_stats.remove(arm);
    }

    fn reset(&mut self) {
        self.arm_stats.clear();
    }

    fn update_with_context(
        &mut self,
        _decisions: &[A],
        _contexts: Option<&[Vec<f64>]>,
        _rewards: &[f64],
    ) -> PolicyResult<()> {
        Err(PolicyError::ContextNotSupported)
    }

    fn select_with_context(
        &self,
        _arms: &IndexSet<A>,
        _context: Option<&[f64]>,
        _rng: &mut dyn rand::RngCore,
    ) -> PolicyResult<Option<A>> {
        Err(PolicyError::ContextNotSupported)
    }

    fn expectations_with_context(
        &self,
        _arms: &IndexSet<A>,
        _context: Option<&[f64]>,
    ) -> PolicyResult<HashMap<A, f64>> {
        Err(PolicyError::ContextNotSupported)
    }
}

impl<A> EpsilonGreedy<A>
where
    A: Clone + Eq + Hash,
{
    /// Find the arm with highest average reward
    fn find_best_arm(&self, arms: &IndexSet<A>) -> Option<A> {
        arms.iter()
            .max_by(|a, b| {
                let reward_a = self.arm_stats.get(*a).map_or(0.0, |s| s.average_reward());
                let reward_b = self.arm_stats.get(*b).map_or(0.0, |s| s.average_reward());
                reward_a
                    .partial_cmp(&reward_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_epsilon_greedy_pure_exploration() {
        let policy = EpsilonGreedy::new(1.0);
        let mut arms = IndexSet::new();
        arms.insert("a");
        arms.insert("b");
        arms.insert("c");
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Should return one of the arms randomly
        let choice = Policy::select(&policy, &arms, &mut rng as &mut dyn rand::RngCore).unwrap();
        assert!(arms.contains(&choice));

        // Expectations should be uniform
        let expectations = policy.expectations(&arms);
        for (_arm, prob) in expectations {
            assert!((prob - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_epsilon_greedy_pure_exploitation() {
        let mut policy = EpsilonGreedy::new(0.0);
        let mut arms = IndexSet::new();
        arms.insert(1);
        arms.insert(2);
        arms.insert(3);

        // Train with some data - arm 2 has highest average
        policy.update(&[1, 2, 3, 2], &[0.5, 1.0, 0.3, 0.8]);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Should always pick arm 2 (highest average reward)
        for _ in 0..10 {
            let choice =
                Policy::select(&policy, &arms, &mut rng as &mut dyn rand::RngCore).unwrap();
            assert_eq!(choice, 2);
        }

        // Expectations should give 100% to best arm
        let expectations = policy.expectations(&arms);
        assert!((expectations[&2] - 1.0).abs() < 1e-10);
        assert!((expectations[&1] - 0.0).abs() < 1e-10);
        assert!((expectations[&3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy_mixed() {
        let mut policy = EpsilonGreedy::new(0.3);
        let mut arms = IndexSet::new();
        arms.insert("x");
        arms.insert("y");
        arms.insert("z");

        // Train with some data - "y" has highest average
        policy.update(&["x", "y", "z", "y"], &[0.4, 0.9, 0.2, 0.8]);

        // Check expectations
        let expectations = policy.expectations(&arms);

        // "y" should get 70% + 10% = 80%
        assert!((expectations[&"y"] - 0.8).abs() < 1e-10);
        // Others should get 10% each
        assert!((expectations[&"x"] - 0.1).abs() < 1e-10);
        assert!((expectations[&"z"] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy_reset() {
        let mut policy = EpsilonGreedy::new(0.5);

        // Train with some data
        policy.update(&[1, 2, 3], &[0.5, 0.8, 0.3]);

        // Reset arm 2
        policy.reset_arm(&2);

        // Arm 2 should now have default stats (0.0 average)
        let mut arms = IndexSet::new();
        arms.insert(1);
        arms.insert(2);
        arms.insert(3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // With epsilon=0.5 and arm 1 being best, it should be selected often
        let mut count_1 = 0;
        for _ in 0..100 {
            if Policy::select(&policy, &arms, &mut rng as &mut dyn rand::RngCore) == Some(1) {
                count_1 += 1;
            }
        }

        // Arm 1 should be selected more than random (33%)
        assert!(count_1 > 40);
    }

    #[test]
    fn test_epsilon_greedy_getters_setters() {
        let mut policy = EpsilonGreedy::<i32>::new(0.5);
        assert_eq!(policy.epsilon(), 0.5);

        policy.set_epsilon(0.3);
        assert_eq!(policy.epsilon(), 0.3);
    }

    #[test]
    fn test_arm_stats() {
        let mut policy = EpsilonGreedy::new(0.5);

        // Initially no stats
        assert_eq!(policy.arm_stats(&1), None);

        // After training
        policy.update(&[1, 1, 2], &[0.5, 0.7, 0.3]);

        assert_eq!(policy.arm_stats(&1), Some((2, 0.6)));
        assert_eq!(policy.arm_stats(&2), Some((1, 0.3)));
        assert_eq!(policy.arm_stats(&3), None);
    }
}
