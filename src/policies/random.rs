use super::Policy;
use rand::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

/// Random selection policy - selects arms uniformly at random
#[derive(Clone, Debug, Default)]
pub struct Random;

impl<A> Policy<A> for Random
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, _decisions: &[A], _rewards: &[f64]) {
        // Random policy doesn't learn from feedback
    }

    fn select(&self, arms: &[A], rng: &mut dyn rand::RngCore) -> Option<A> {
        arms.choose(rng).cloned()
    }

    fn expectations(&self, arms: &[A]) -> HashMap<A, f64> {
        if arms.is_empty() {
            return HashMap::new();
        }

        let prob = 1.0 / arms.len() as f64;
        arms.iter().map(|arm| (arm.clone(), prob)).collect()
    }

    fn reset_arm(&mut self, _arm: &A) {
        // No state to reset
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_select() {
        let policy = Random;
        let arms = vec!["a", "b", "c"];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let choice = Policy::select(&policy, &arms, &mut rng as &mut dyn rand::RngCore).unwrap();
        assert!(arms.contains(&choice));
    }

    #[test]
    fn test_random_expectations() {
        let policy = Random;
        let arms = vec![1, 2, 3];
        let expectations = policy.expectations(&arms);

        assert_eq!(expectations.len(), 3);
        for (_arm, prob) in expectations {
            assert!((prob - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_empty() {
        let policy = Random;
        let arms: Vec<i32> = vec![];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        assert_eq!(
            Policy::select(&policy, &arms, &mut rng as &mut dyn rand::RngCore),
            None
        );
        assert_eq!(policy.expectations(&arms).len(), 0);
    }
}
