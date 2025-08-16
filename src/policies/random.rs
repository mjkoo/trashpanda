use super::Policy;
use indexmap::IndexSet;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Random selection policy - selects arms uniformly at random
#[derive(Clone, Debug, Default)]
pub struct Random;

impl<A> Policy<A, ()> for Random
where
    A: Clone + Eq + Hash,
{
    fn update(&mut self, _decision: &A, _context: (), _reward: f64) {
        // Random policy doesn't learn from feedback
    }

    fn select(&self, arms: &IndexSet<A>, _context: (), rng: &mut dyn rand::RngCore) -> Option<A> {
        if arms.is_empty() {
            return None;
        }

        let idx = rng.random_range(0..arms.len());
        arms.get_index(idx).cloned()
    }

    fn expectations(&self, arms: &IndexSet<A>, _context: ()) -> HashMap<A, f64> {
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
        let policy = Random;
        let mut arms = IndexSet::new();
        arms.insert(1);
        arms.insert(2);
        arms.insert(3);
        let expectations = policy.expectations(&arms, ());

        assert_eq!(expectations.len(), 3);
        for (_arm, prob) in expectations {
            assert!((prob - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_random_empty() {
        let policy = Random;
        let arms: IndexSet<i32> = IndexSet::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        assert_eq!(Policy::select(&policy, &arms, (), &mut rng), None);
        assert_eq!(policy.expectations(&arms, ()).len(), 0);
    }
}
