use super::Policy;
use rand::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

/// Epsilon-greedy policy - explores with probability epsilon, exploits otherwise
#[derive(Clone)]
pub struct EpsilonGreedy<A> {
    epsilon: f64,
    arm_stats: HashMap<A, ArmStats>,
}

// Conditional Debug implementation
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
        Self {
            epsilon,
            arm_stats: HashMap::new(),
        }
    }
}

impl<A> Policy<A> for EpsilonGreedy<A>
where
    A: Clone + Eq + Hash + Send + Sync,
{
    fn update(&mut self, decisions: &[A], rewards: &[f64]) {
        for (arm, reward) in decisions.iter().zip(rewards.iter()) {
            let stats = self.arm_stats.entry(arm.clone()).or_default();
            stats.pulls += 1;
            stats.total_reward += reward;
        }
    }

    fn select(&self, arms: &[A], rng: &mut dyn rand::RngCore) -> Option<A> {
        use rand::Rng;

        if arms.is_empty() {
            return None;
        }

        // Explore with probability epsilon
        if rng.random::<f64>() < self.epsilon {
            // Random selection (exploration)
            arms.choose(rng).cloned()
        } else {
            // Greedy selection (exploitation)
            self.find_best_arm(arms)
        }
    }

    fn expectations(&self, arms: &[A]) -> HashMap<A, f64> {
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
}

impl<A> EpsilonGreedy<A>
where
    A: Clone + Eq + Hash,
{
    /// Finds the arm with the highest average reward from the given arms
    fn find_best_arm(&self, arms: &[A]) -> Option<A> {
        arms.iter()
            .map(|arm| {
                let avg = self
                    .arm_stats
                    .get(arm)
                    .map(|s| s.average_reward())
                    .unwrap_or(0.0);
                (arm, avg)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arm, _)| arm.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_epsilon_greedy_pure_exploration() {
        let policy = EpsilonGreedy::new(1.0);
        let arms = vec!["a", "b", "c"];
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
        let arms = vec![1, 2, 3];

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
        let arms = vec!["x", "y", "z"];

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
        let arms = vec![1, 2, 3];
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
}
