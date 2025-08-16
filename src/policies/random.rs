use super::Policy;
use indexmap::IndexSet;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Random selection policy - selects arms uniformly at random
///
/// While the selection is random, this policy still tracks reward statistics
/// to provide meaningful expected rewards in the expectations() method.
#[derive(Clone, Debug)]
pub struct Random<A> {
    /// Statistics for each arm (for tracking expected rewards)
    arm_stats: HashMap<A, ArmStats>,
}

impl<A> Default for Random<A> {
    fn default() -> Self {
        Self {
            arm_stats: HashMap::new(),
        }
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

impl<A> Policy<A, ()> for Random<A>
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, decision: &A, _context: (), reward: f64) {
        // Track rewards for expectations, even though selection remains random
        let stats = self.arm_stats.entry(decision.clone()).or_default();
        stats.pulls += 1;
        stats.total_reward += reward;
    }

    fn select(&self, arms: &IndexSet<A>, _context: (), rng: &mut dyn rand::RngCore) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        let idx = rng.random_range(0..arms.len());
        arms.get_index(idx).cloned()
    }

    fn expectations(
        &self,
        arms: &IndexSet<A>,
        _context: (),
        _rng: &mut dyn rand::RngCore,
    ) -> HashMap<A, f64> {
        // Return the actual average rewards observed for each arm
        // Even though selection is random, we can still report expected rewards
        let mut expectations = HashMap::new();
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_select() {
        let policy = Random::default();
        let mut arms = IndexSet::new();
        arms.insert("a");
        arms.insert("b");
        arms.insert("c");
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let choice = Policy::select(&policy, &arms, (), &mut rng).unwrap();
        assert!(arms.contains(&choice));
    }

    #[test]
    fn test_random_expectations() {
        let mut policy = Random::default();
        let mut arms = IndexSet::new();
        arms.insert(1);
        arms.insert(2);
        arms.insert(3);

        // Before any training, all expectations should be 0.0
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);
        assert_eq!(expectations.len(), 3);
        for expected_reward in expectations.values() {
            assert_eq!(*expected_reward, 0.0);
        }

        // After training, expectations should reflect actual average rewards
        policy.update(&1, (), 0.5);
        policy.update(&1, (), 0.7);
        policy.update(&2, (), 0.3);
        policy.update(&3, (), 0.9);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let expectations = policy.expectations(&arms, (), &mut rng);
        assert_eq!(expectations.len(), 3);
        assert!((expectations[&1] - 0.6).abs() < 1e-10); // (0.5 + 0.7) / 2
        assert!((expectations[&2] - 0.3).abs() < 1e-10);
        assert!((expectations[&3] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_random_empty() {
        let policy = Random::default();
        let arms: IndexSet<i32> = IndexSet::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        assert_eq!(Policy::select(&policy, &arms, (), &mut rng), None);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        assert_eq!(policy.expectations(&arms, (), &mut rng).len(), 0);
    }
}
