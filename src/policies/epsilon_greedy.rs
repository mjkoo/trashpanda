use super::Policy;
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
    #[must_use]
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
impl<A> Policy<A, ()> for EpsilonGreedy<A>
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, decision: &A, _context: (), reward: f64) {
        let stats = self.arm_stats.entry(decision.clone()).or_default();
        stats.pulls += 1;
        stats.total_reward += reward;
    }

    fn select(&self, arms: &IndexSet<A>, _context: (), rng: &mut dyn rand::RngCore) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        // Explore with probability epsilon
        let r: f64 = rng.random_range(0.0..1.0);
        if r < self.epsilon {
            // Random selection (exploration)
            let idx = rng.random_range(0..arms.len());
            arms.get_index(idx).cloned()
        } else {
            // Select arm with highest average reward (exploitation)
            self.find_best_arm(arms)
        }
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        _context: (),
        _rng: &mut dyn rand::RngCore,
    ) -> HashMap<A, f64> {
        let mut expectations = HashMap::new();

        // Return the average reward (expectation) for each arm
        // Arms without data have an expectation of 0.0
        for arm in arms {
            let expected_reward = self
                .arm_stats
                .get(arm)
                .map_or(0.0, |stats| stats.average_reward());
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
        let choice = Policy::select(&policy, &arms, (), &mut rng).unwrap();
        assert!(arms.contains(&choice));

        // All arms should have 0.0 expected reward (no training data)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);
        for (_arm, expected_reward) in expectations {
            assert_eq!(expected_reward, 0.0);
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
        policy.update(&1, (), 0.5);
        policy.update(&2, (), 1.0);
        policy.update(&3, (), 0.3);
        policy.update(&2, (), 0.8);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Should always pick arm 2 (highest average reward)
        for _ in 0..10 {
            let choice = Policy::select(&policy, &arms, (), &mut rng).unwrap();
            assert_eq!(choice, 2);
        }

        // Expectations should reflect actual average rewards
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);
        assert!((expectations[&2] - 0.9).abs() < 1e-10); // (1.0 + 0.8) / 2 = 0.9
        assert!((expectations[&1] - 0.5).abs() < 1e-10); // 0.5 / 1 = 0.5
        assert!((expectations[&3] - 0.3).abs() < 1e-10); // 0.3 / 1 = 0.3
    }

    #[test]
    fn test_epsilon_greedy_mixed() {
        let mut policy = EpsilonGreedy::new(0.3);
        let mut arms = IndexSet::new();
        arms.insert("x");
        arms.insert("y");
        arms.insert("z");

        // Train with some data - "y" has highest average
        policy.update(&"x", (), 0.4);
        policy.update(&"y", (), 0.9);
        policy.update(&"z", (), 0.2);
        policy.update(&"y", (), 0.8);

        // Check expectations - should reflect actual average rewards
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);

        // "y" should have average of (0.9 + 0.8) / 2 = 0.85
        assert!((expectations[&"y"] - 0.85).abs() < 1e-10);
        // "x" should have 0.4 / 1 = 0.4
        assert!((expectations[&"x"] - 0.4).abs() < 1e-10);
        // "z" should have 0.2 / 1 = 0.2
        assert!((expectations[&"z"] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy_reset() {
        let mut policy = EpsilonGreedy::new(0.5);

        // Train with some data
        policy.update(&1, (), 0.5);
        policy.update(&2, (), 0.8);
        policy.update(&3, (), 0.3);

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
            if Policy::select(&policy, &arms, (), &mut rng) == Some(1) {
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
        policy.update(&1, (), 0.5);
        policy.update(&1, (), 0.7);
        policy.update(&2, (), 0.3);

        assert_eq!(policy.arm_stats(&1), Some((2, 0.6)));
        assert_eq!(policy.arm_stats(&2), Some((1, 0.3)));
        assert_eq!(policy.arm_stats(&3), None);
    }
}
