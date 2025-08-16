use std::collections::HashMap;
use std::hash::Hash;

use indexmap::IndexSet;

use crate::policy::Policy;

/// Upper Confidence Bound (UCB1) policy
///
/// This policy balances exploration and exploitation by selecting arms based on
/// their upper confidence bounds. Arms with higher uncertainty or higher average
/// rewards are more likely to be selected.
#[derive(Clone)]
pub struct Ucb<A> {
    /// Confidence parameter (typically sqrt(2) ≈ 1.414)
    alpha: f64,
    /// Statistics for each arm
    arm_stats: HashMap<A, ArmStats>,
    /// Total number of rounds played
    total_rounds: usize,
}

impl<A> std::fmt::Debug for Ucb<A>
where
    A: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ucb")
            .field("alpha", &self.alpha)
            .field("total_rounds", &self.total_rounds)
            .field("arm_stats", &self.arm_stats)
            .finish()
    }
}

#[derive(Clone, Debug, Default)]
struct ArmStats {
    pulls: usize,
    total_reward: f64,
    average_reward: f64,
}

impl ArmStats {
    fn average_reward(&self) -> f64 {
        if self.pulls == 0 {
            0.0
        } else {
            self.total_reward / self.pulls as f64
        }
    }

    fn ucb_score(&self, total_rounds: usize, alpha: f64) -> f64 {
        if self.pulls == 0 {
            // Unpulled arms have infinite UCB score (explore first)
            f64::INFINITY
        } else {
            let exploitation = self.average_reward();
            let exploration =
                alpha * ((2.0 * (total_rounds as f64).ln()) / self.pulls as f64).sqrt();
            exploitation + exploration
        }
    }
}

impl<A> Ucb<A>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new UCB1 policy with the given confidence parameter
    ///
    /// # Arguments
    /// * `alpha` - Confidence parameter (typically sqrt(2) ≈ 1.414)
    ///   Higher values encourage more exploration
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        Self {
            alpha,
            arm_stats: HashMap::new(),
            total_rounds: 0,
        }
    }

    /// Gets the confidence parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Sets the confidence parameter
    pub fn set_alpha(&mut self, alpha: f64) {
        assert!(alpha > 0.0, "alpha must be positive");
        self.alpha = alpha;
    }

    /// Gets the statistics for a specific arm
    pub fn arm_stats(&self, arm: &A) -> Option<(usize, f64, f64)> {
        self.arm_stats.get(arm).map(|s| {
            (
                s.pulls,
                s.average_reward(),
                s.ucb_score(self.total_rounds, self.alpha),
            )
        })
    }

    /// Gets the total number of rounds played
    pub fn total_rounds(&self) -> usize {
        self.total_rounds
    }
}

impl<A> Policy<A, ()> for Ucb<A>
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, decision: &A, _context: (), reward: f64) {
        let stats = self.arm_stats.entry(decision.clone()).or_default();
        stats.pulls += 1;
        stats.total_reward += reward;
        stats.average_reward = stats.total_reward / stats.pulls as f64;
        self.total_rounds += 1;
    }

    fn select(&self, arms: &IndexSet<A>, _context: (), _rng: &mut dyn rand::RngCore) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        // Select arm with highest UCB score
        arms.iter()
            .max_by(|a, b| {
                let score_a = self.arm_stats.get(*a).map_or(f64::INFINITY, |s| {
                    s.ucb_score(self.total_rounds, self.alpha)
                });
                let score_b = self.arm_stats.get(*b).map_or(f64::INFINITY, |s| {
                    s.ucb_score(self.total_rounds, self.alpha)
                });

                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
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
                .map_or(0.0, |stats| stats.average_reward);
            expectations.insert(arm.clone(), expected_reward);
        }

        expectations
    }

    fn reset_arm(&mut self, arm: &A) {
        self.arm_stats.remove(arm);
    }

    fn reset(&mut self) {
        self.arm_stats.clear();
        self.total_rounds = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;
    use rand::SeedableRng;

    #[test]
    fn test_ucb_explores_unpulled_arms_first() {
        let policy = Ucb::new(1.414);
        let mut arms = IndexSet::new();
        arms.insert("a");
        arms.insert("b");
        arms.insert("c");

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Should select an unpulled arm (any of them)
        let choice = Policy::select(&policy, &arms, (), &mut rng).unwrap();
        assert!(arms.contains(&choice));

        // All unpulled arms should have 0.0 expected reward (no data)
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);
        for (_arm, expected_reward) in expectations {
            assert_eq!(expected_reward, 0.0);
        }
    }

    #[test]
    fn test_ucb_balances_exploration_exploitation() {
        let mut policy = Ucb::new(1.414);
        let mut arms = IndexSet::new();
        arms.insert(1);
        arms.insert(2);
        arms.insert(3);

        // Train with some data - arm 2 has highest average but arm 3 is less explored
        policy.update(&1, (), 0.5);
        policy.update(&1, (), 0.4);
        policy.update(&1, (), 0.6);
        policy.update(&2, (), 0.9);
        policy.update(&2, (), 0.8);
        policy.update(&3, (), 0.7);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Should balance between high reward (arm 2) and exploration (arm 3)
        let choice = Policy::select(&policy, &arms, (), &mut rng).unwrap();
        // The exact choice depends on UCB calculation
        assert!([2, 3].contains(&choice));
    }

    #[test]
    fn test_ucb_confidence_parameter() {
        let mut policy = Ucb::<i32>::new(1.0);
        assert_eq!(policy.alpha(), 1.0);

        policy.set_alpha(2.0);
        assert_eq!(policy.alpha(), 2.0);
    }

    #[test]
    fn test_ucb_deterministic_selection() {
        let mut policy = Ucb::new(1.414);
        let mut arms = IndexSet::new();
        arms.insert("x");
        arms.insert("y");
        arms.insert("z");

        // Train with clear winner
        policy.update(&"x", (), 0.1);
        policy.update(&"y", (), 0.5);
        policy.update(&"z", (), 0.9);
        policy.update(&"x", (), 0.2);
        policy.update(&"y", (), 0.6);
        policy.update(&"z", (), 0.8);

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(1);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(999);

        // UCB is deterministic - should select same arm regardless of RNG
        let choice1 = Policy::select(&policy, &arms, (), &mut rng1).unwrap();
        let choice2 = Policy::select(&policy, &arms, (), &mut rng2).unwrap();
        assert_eq!(choice1, choice2);
    }

    #[test]
    fn test_ucb_arm_stats() {
        let mut policy = Ucb::new(1.414);

        // Initially no stats
        assert_eq!(policy.arm_stats(&1), None);

        // After training
        policy.update(&1, (), 0.5);
        policy.update(&1, (), 0.7);
        policy.update(&2, (), 0.3);

        let stats = policy.arm_stats(&1).unwrap();
        assert_eq!(stats.0, 2); // pulls
        assert!(abs_diff_eq!(stats.1, 0.6)); // average reward
        assert!(stats.2 > 0.6); // UCB score should be higher than average

        assert_eq!(policy.total_rounds(), 3);
    }

    #[test]
    fn test_ucb_reset() {
        let mut policy = Ucb::new(1.414);

        policy.update(&1, (), 0.5);
        policy.update(&2, (), 0.8);
        policy.update(&3, (), 0.3);
        assert_eq!(policy.total_rounds(), 3);

        // Reset specific arm
        policy.reset_arm(&2);
        assert_eq!(policy.arm_stats(&2), None);
        assert!(policy.arm_stats(&1).is_some());

        // Full reset
        policy.reset();
        assert_eq!(policy.total_rounds(), 0);
        assert_eq!(policy.arm_stats(&1), None);
    }
}
