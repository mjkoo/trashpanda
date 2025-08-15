use crate::policies::{EpsilonGreedy, Policy, Random};
use crate::{BanditError, Result};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// Learning policy configuration.
#[derive(Clone, Debug)]
pub enum LearningPolicy {
    /// Epsilon-greedy policy with specified exploration rate.
    EpsilonGreedy { epsilon: f64 },
    /// Upper Confidence Bound with specified confidence parameter.
    Ucb { alpha: f64 },
    /// Thompson Sampling for binary rewards.
    ThompsonSampling,
    /// Random selection baseline.
    Random,
}

/// Metadata associated with each arm.
#[derive(Clone, Debug, Default)]
pub struct ArmMetadata {
    /// Whether this arm has been trained on any data.
    pub is_trained: bool,
    /// Number of times this arm has been pulled.
    pub pull_count: usize,
    /// Total reward accumulated for this arm.
    pub total_reward: f64,
    /// Sum of squared rewards (for variance calculation).
    pub sum_squared_reward: f64,
}

/// Generic multi-armed bandit implementation.
pub struct Bandit<A, P> {
    /// Available arms.
    arms: Vec<A>,
    /// Set of arms for O(1) membership checking.
    arm_set: HashSet<A>,
    /// Metadata for each arm.
    arm_metadata: HashMap<A, ArmMetadata>,
    /// Policy implementation.
    policy: P,
}

impl<A, P> std::fmt::Debug for Bandit<A, P>
where
    A: Clone + Eq + Hash + Debug,
    P: Policy<A> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bandit")
            .field("arms", &self.arms)
            .field("n_arms", &self.n_arms())
            .field("policy", &self.policy)
            .finish()
    }
}

impl<A, P> Bandit<A, P>
where
    A: Clone + Eq + Hash,
    P: Policy<A>,
{
    /// Creates a new bandit with the given arms and policy.
    pub fn new<I>(arms: I, policy: P) -> Result<Self>
    where
        I: IntoIterator<Item = A>,
    {
        let arms: Vec<A> = arms.into_iter().collect();

        if arms.is_empty() {
            return Err(BanditError::BuilderError {
                message: "at least one arm must be specified".to_string(),
            });
        }

        // Check for duplicate arms
        let arm_set: HashSet<_> = arms.iter().cloned().collect();
        if arm_set.len() != arms.len() {
            return Err(BanditError::BuilderError {
                message: "duplicate arms provided".to_string(),
            });
        }

        // Initialize metadata
        let mut arm_metadata = HashMap::new();
        for arm in &arms {
            arm_metadata.insert(arm.clone(), ArmMetadata::default());
        }

        Ok(Bandit {
            arms,
            arm_set,
            arm_metadata,
            policy,
        })
    }

    /// Returns a slice of all arms in the bandit.
    pub fn arms(&self) -> &[A] {
        &self.arms
    }

    /// Returns the number of arms.
    pub fn n_arms(&self) -> usize {
        self.arms.len()
    }

    /// Checks if an arm exists in the bandit.
    pub fn has_arm(&self, arm: &A) -> bool {
        self.arm_set.contains(arm)
    }

    /// Returns a reference to the policy.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Returns a mutable reference to the policy.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Adds a new arm to the bandit.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::policies::Random;
    /// use trashpanda::Bandit;
    ///
    /// let mut bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();
    /// bandit.add_arm(4).unwrap();
    /// assert_eq!(bandit.n_arms(), 4);
    /// ```
    pub fn add_arm(&mut self, arm: A) -> Result<()> {
        if self.has_arm(&arm) {
            return Err(BanditError::ArmAlreadyExists);
        }

        self.arms.push(arm.clone());
        self.arm_set.insert(arm.clone());
        self.arm_metadata
            .insert(arm.clone(), ArmMetadata::default());

        // Reset policy stats for this arm
        self.policy.reset_arm(&arm);

        Ok(())
    }

    /// Removes an arm from the bandit.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::policies::Random;
    /// use trashpanda::Bandit;
    ///
    /// let mut bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();
    /// bandit.remove_arm(&2).unwrap();
    /// assert_eq!(bandit.n_arms(), 2);
    /// ```
    pub fn remove_arm(&mut self, arm: &A) -> Result<()> {
        if !self.has_arm(arm) {
            return Err(BanditError::ArmNotFound);
        }

        self.arms.retain(|a| a != arm);
        self.arm_set.remove(arm);
        self.arm_metadata.remove(arm);

        Ok(())
    }

    /// Trains the bandit on historical data.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::policies::EpsilonGreedy;
    /// use trashpanda::Bandit;
    ///
    /// let mut bandit = Bandit::new(
    ///     vec!["a", "b", "c"],
    ///     EpsilonGreedy::new(0.1)
    /// ).unwrap();
    ///
    /// let decisions = vec!["a", "b", "a"];
    /// let rewards = vec![1.0, 0.5, 0.8];
    /// bandit.fit(&decisions, &rewards).unwrap();
    /// ```
    pub fn fit(&mut self, decisions: &[A], rewards: &[f64]) -> Result<()> {
        if decisions.len() != rewards.len() {
            return Err(BanditError::DimensionMismatch {
                message: format!(
                    "decisions and rewards must have same length: {} != {}",
                    decisions.len(),
                    rewards.len()
                ),
            });
        }

        // Validate all decisions are valid arms
        for decision in decisions {
            if !self.has_arm(decision) {
                return Err(BanditError::ArmNotFound);
            }
        }

        // Update metadata
        for (arm, reward) in decisions.iter().zip(rewards.iter()) {
            if let Some(metadata) = self.arm_metadata.get_mut(arm) {
                metadata.is_trained = true;
                metadata.pull_count += 1;
                metadata.total_reward += reward;
                metadata.sum_squared_reward += reward * reward;
            }
        }

        // Update policy
        self.policy.update(decisions, rewards);

        Ok(())
    }

    /// Incrementally trains the bandit on new data.
    ///
    /// This is equivalent to `fit` and exists for compatibility with scikit-learn conventions.
    pub fn partial_fit(&mut self, decisions: &[A], rewards: &[f64]) -> Result<()> {
        self.fit(decisions, rewards)
    }

    /// Predicts the best arm to pull.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::policies::Random;
    /// use trashpanda::Bandit;
    ///
    /// let bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();
    /// let choice = bandit.predict().unwrap();
    /// assert!([1, 2, 3].contains(&choice));
    /// ```
    pub fn predict(&self) -> Result<A> {
        self.predict_with_rng(&mut rand::rng())
    }

    /// Predicts the best arm to pull using a specific random number generator.
    pub fn predict_with_rng<R: Rng>(&self, rng: &mut R) -> Result<A> {
        self.policy
            .select(&self.arms, rng as &mut dyn rand::RngCore)
            .ok_or(BanditError::NoArmsAvailable)
    }

    /// Returns the expected reward probability for each arm.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::policies::EpsilonGreedy;
    /// use trashpanda::Bandit;
    ///
    /// let mut bandit = Bandit::new(
    ///     vec!["a", "b", "c"],
    ///     EpsilonGreedy::new(0.2)
    /// ).unwrap();
    ///
    /// bandit.fit(&["a", "b", "c"], &[0.8, 0.5, 0.3]).unwrap();
    /// let expectations = bandit.predict_expectations().unwrap();
    /// ```
    pub fn predict_expectations(&self) -> Result<HashMap<A, f64>> {
        Ok(self.policy.expectations(&self.arms))
    }
}

// Convenience builder for the common use case with LearningPolicy enum
impl<A> Bandit<A, EpsilonGreedy<A>>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new epsilon-greedy bandit.
    pub fn epsilon_greedy<I>(arms: I, epsilon: f64) -> Result<Self>
    where
        I: IntoIterator<Item = A>,
    {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(BanditError::InvalidParameter {
                message: format!("epsilon must be between 0 and 1, got {}", epsilon),
            });
        }
        Self::new(arms, EpsilonGreedy::new(epsilon))
    }
}

impl<A> Bandit<A, Random>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new random selection bandit.
    pub fn random<I>(arms: I) -> Result<Self>
    where
        I: IntoIterator<Item = A>,
    {
        Self::new(arms, Random)
    }
}

/// Builder for constructing a Bandit instance with the legacy API.
pub struct BanditBuilder<A> {
    arms: Option<Vec<A>>,
    policy: Option<LearningPolicy>,
}

impl<A> BanditBuilder<A>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            arms: None,
            policy: None,
        }
    }

    /// Sets the arms for the bandit.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::{Bandit, LearningPolicy};
    ///
    /// let builder = Bandit::<String, ()>::builder()
    ///     .arms(vec!["a".to_string(), "b".to_string()]);
    /// ```
    pub fn arms<I>(mut self, arms: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        self.arms = Some(arms.into_iter().collect());
        self
    }

    /// Sets the learning policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::{Bandit, LearningPolicy};
    ///
    /// let builder = Bandit::<i32, ()>::builder()
    ///     .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 });
    /// ```
    pub fn policy(mut self, policy: LearningPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Builds the Bandit instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No arms were specified
    /// - No policy was specified  
    /// - Duplicate arms were provided
    /// - Zero arms were provided
    /// - Invalid policy parameters
    pub fn build<P>(self) -> Result<Bandit<A, P>>
    where
        P: BuildablePolicy<A>,
    {
        let arms = self.arms.ok_or_else(|| BanditError::BuilderError {
            message: "arms must be specified".to_string(),
        })?;

        let policy_type = self.policy.ok_or_else(|| BanditError::BuilderError {
            message: "policy must be specified".to_string(),
        })?;

        P::build_from_config(arms, policy_type)
    }
}

impl<A> Default for BanditBuilder<A>
where
    A: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

// Helper trait for building policies from LearningPolicy enum
pub trait BuildablePolicy<A>: Policy<A> + Sized {
    fn build_from_config(arms: Vec<A>, config: LearningPolicy) -> Result<Bandit<A, Self>>;
}

impl<A> BuildablePolicy<A> for EpsilonGreedy<A>
where
    A: Clone + Eq + Hash,
{
    fn build_from_config(arms: Vec<A>, config: LearningPolicy) -> Result<Bandit<A, Self>> {
        match config {
            LearningPolicy::EpsilonGreedy { epsilon } => {
                if !(0.0..=1.0).contains(&epsilon) {
                    return Err(BanditError::InvalidParameter {
                        message: format!("epsilon must be between 0 and 1, got {}", epsilon),
                    });
                }
                Bandit::new(arms, EpsilonGreedy::new(epsilon))
            }
            _ => Err(BanditError::BuilderError {
                message: "policy type mismatch".to_string(),
            }),
        }
    }
}

impl<A> BuildablePolicy<A> for Random
where
    A: Clone + Eq + Hash,
{
    fn build_from_config(arms: Vec<A>, config: LearningPolicy) -> Result<Bandit<A, Self>> {
        match config {
            LearningPolicy::Random => Bandit::new(arms, Random),
            _ => Err(BanditError::BuilderError {
                message: "policy type mismatch".to_string(),
            }),
        }
    }
}

// Convenience method for builder on Bandit when no policy type is specified
impl<A> Bandit<A, ()>
where
    A: Clone + Eq + Hash,
{
    /// Creates a new bandit builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::{Bandit, LearningPolicy};
    ///
    /// let bandit = Bandit::builder()
    ///     .arms(vec![1, 2, 3])
    ///     .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
    ///     .build::<trashpanda::policies::EpsilonGreedy<_>>()
    ///     .unwrap();
    /// ```
    pub fn builder() -> BanditBuilder<A> {
        BanditBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_construction() {
        // Test with EpsilonGreedy
        let bandit = Bandit::new(vec!["a", "b", "c"], EpsilonGreedy::new(0.1)).unwrap();
        assert_eq!(bandit.n_arms(), 3);

        // Test with Random
        let bandit = Bandit::new(vec![1, 2, 3], Random).unwrap();
        assert_eq!(bandit.n_arms(), 3);
    }

    #[test]
    fn test_convenience_constructors() {
        // Test epsilon_greedy constructor
        let bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.1).unwrap();
        assert_eq!(bandit.n_arms(), 3);

        // Test random constructor
        let bandit = Bandit::random(vec!["a", "b", "c"]).unwrap();
        assert_eq!(bandit.n_arms(), 3);

        // Test invalid epsilon
        let result = Bandit::epsilon_greedy(vec![1, 2, 3], 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_explicit_type() {
        let bandit = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
            .build::<EpsilonGreedy<_>>()
            .unwrap();

        assert_eq!(bandit.n_arms(), 3);
    }

    #[test]
    fn test_builder_with_random() {
        let bandit = Bandit::builder()
            .arms(vec!["a", "b", "c"])
            .policy(LearningPolicy::Random)
            .build::<Random>()
            .unwrap();

        assert_eq!(bandit.n_arms(), 3);
    }

    #[test]
    fn test_builder_errors() {
        // No arms
        let result = Bandit::<i32, ()>::builder()
            .policy(LearningPolicy::Random)
            .build::<Random>();
        assert!(result.is_err());

        // No policy
        let result = Bandit::builder().arms(vec![1, 2, 3]).build::<Random>();
        assert!(result.is_err());

        // Empty arms
        let result = Bandit::<i32, ()>::builder()
            .arms(vec![])
            .policy(LearningPolicy::Random)
            .build::<Random>();
        assert!(result.is_err());

        // Duplicate arms
        let result = Bandit::builder()
            .arms(vec![1, 2, 2, 3])
            .policy(LearningPolicy::Random)
            .build::<Random>();
        assert!(result.is_err());

        // Invalid epsilon
        let result = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::EpsilonGreedy { epsilon: -0.1 })
            .build::<EpsilonGreedy<_>>();
        assert!(result.is_err());
    }

    #[test]
    fn test_add_remove_arms() {
        let mut bandit = Bandit::random(vec![1, 2, 3]).unwrap();

        // Add new arm
        assert!(bandit.add_arm(4).is_ok());
        assert_eq!(bandit.n_arms(), 4);
        assert!(bandit.has_arm(&4));

        // Try to add duplicate
        assert!(bandit.add_arm(4).is_err());

        // Remove arm
        assert!(bandit.remove_arm(&2).is_ok());
        assert_eq!(bandit.n_arms(), 3);
        assert!(!bandit.has_arm(&2));

        // Try to remove non-existent
        assert!(bandit.remove_arm(&10).is_err());
    }

    #[test]
    fn test_fit_validates_arms() {
        let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.1).unwrap();

        // Valid decisions
        assert!(bandit.fit(&[1, 2, 3], &[0.5, 0.8, 0.3]).is_ok());

        // Invalid arm
        assert!(bandit.fit(&[1, 2, 4], &[0.5, 0.8, 0.3]).is_err());

        // Mismatched lengths
        assert!(bandit.fit(&[1, 2], &[0.5, 0.8, 0.3]).is_err());
    }

    #[test]
    fn test_policy_access() {
        let mut bandit = Bandit::epsilon_greedy(vec![1, 2, 3], 0.5).unwrap();

        // Access policy
        assert_eq!(bandit.policy().epsilon(), 0.5);

        // Mutate policy
        bandit.policy_mut().set_epsilon(0.3);
        assert_eq!(bandit.policy().epsilon(), 0.3);
    }
}
