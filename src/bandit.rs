use crate::policies::{EpsilonGreedy, Policy, Random};
use crate::{BanditError, Result};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

// Note on trait bounds:
// - Clone + Eq + Hash: Required for HashMap storage and arm comparison
// - Send + Sync: Required because we store policies as trait objects (Box<dyn Policy<A>>)
// - Debug: Optional, only required for debug formatting
// - Display: Not required, we use generic error messages instead

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

/// Multi-armed bandit implementation.
pub struct Bandit<A>
where
    A: Clone + Eq + Hash + Send + Sync,
{
    /// Available arms.
    arms: Vec<A>,
    /// Set of arms for O(1) membership checking.
    arm_set: HashSet<A>,
    /// Metadata for each arm.
    arm_metadata: HashMap<A, ArmMetadata>,
    /// Learning policy.
    #[allow(dead_code)] // Will be used when we add policy-specific behaviors
    policy_type: LearningPolicy,
    /// Policy implementation.
    policy: Box<dyn Policy<A> + Send + Sync + 'static>,
}

// Conditional Debug implementation - only available when A implements Debug
impl<A> std::fmt::Debug for Bandit<A>
where
    A: Clone + Eq + Hash + Send + Sync + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bandit")
            .field("arms", &self.arms)
            .field("n_arms", &self.n_arms())
            .field("policy_type", &self.policy_type)
            .finish()
    }
}

impl<A> Bandit<A>
where
    A: Clone + Eq + Hash + Send + Sync,
{
    /// Creates a new bandit builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::{Bandit, LearningPolicy};
    ///
    /// let bandit = Bandit::builder()
    ///     .arms(vec!["red", "blue", "green"])
    ///     .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> BanditBuilder<A> {
        BanditBuilder::new()
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

    /// Adds a new arm to the bandit.
    ///
    /// # Errors
    ///
    /// Returns `BanditError::ArmAlreadyExists` if the arm already exists.
    pub fn add_arm(&mut self, arm: A) -> Result<()> {
        if self.has_arm(&arm) {
            return Err(BanditError::ArmAlreadyExists);
        }

        self.arms.push(arm.clone());
        self.arm_set.insert(arm.clone());
        self.arm_metadata
            .insert(arm.clone(), ArmMetadata::default());

        // Reset policy stats for this arm in case it was previously removed
        self.policy.reset_arm(&arm);

        Ok(())
    }

    /// Removes an arm from the bandit.
    ///
    /// # Errors
    ///
    /// Returns `BanditError::ArmNotFound` if the arm doesn't exist.
    pub fn remove_arm(&mut self, arm: &A) -> Result<()> {
        if !self.has_arm(arm) {
            return Err(BanditError::ArmNotFound);
        }

        self.arms.retain(|a| a != arm);
        self.arm_set.remove(arm);
        self.arm_metadata.remove(arm);

        // Note: We don't reset policy stats here in case the arm is re-added later
        // The policy can decide how to handle missing arms

        Ok(())
    }

    /// Trains the bandit on historical data.
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
    pub fn partial_fit(&mut self, decisions: &[A], rewards: &[f64]) -> Result<()> {
        self.fit(decisions, rewards)
    }

    /// Predicts the best arm to pull using the default RNG.
    pub fn predict(&self) -> Result<A> {
        self.predict_with_rng(&mut rand::rng())
    }

    /// Predicts the best arm to pull using a provided RNG.
    pub fn predict_with_rng<R: Rng>(&self, rng: &mut R) -> Result<A> {
        self.policy
            .select(&self.arms, rng as &mut dyn rand::RngCore)
            .ok_or(BanditError::NoArmsAvailable)
    }

    /// Returns the expected reward for each arm.
    pub fn predict_expectations(&self) -> Result<HashMap<A, f64>> {
        Ok(self.policy.expectations(&self.arms))
    }
}

/// Builder for constructing a `Bandit`.
pub struct BanditBuilder<A>
where
    A: Clone + Eq + Hash + Send + Sync,
{
    arms: Option<Vec<A>>,
    policy: Option<LearningPolicy>,
}

impl<A> BanditBuilder<A>
where
    A: Clone + Eq + Hash + Send + Sync,
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
    /// use trashpanda::Bandit;
    ///
    /// let builder = Bandit::<i32>::builder()
    ///     .arms(vec![1, 2, 3]);
    /// ```
    pub fn arms<I>(mut self, arms: I) -> Self
    where
        I: IntoIterator<Item = A>,
    {
        self.arms = Some(arms.into_iter().collect());
        self
    }

    /// Sets the learning policy.
    pub fn policy(mut self, policy: LearningPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Builds the bandit.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No arms were specified
    /// - No policy was specified
    /// - Duplicate arms were provided
    /// - Invalid policy parameters
    ///
    /// # Note
    /// Requires A: 'static because we store policies as trait objects.
    /// This is satisfied by most common types (String, i32, etc.)
    pub fn build(self) -> Result<Bandit<A>>
    where
        A: 'static,
    {
        let arms = self.arms.ok_or_else(|| BanditError::BuilderError {
            message: "arms must be specified".to_string(),
        })?;

        if arms.is_empty() {
            return Err(BanditError::BuilderError {
                message: "at least one arm must be specified".to_string(),
            });
        }

        let policy_type = self.policy.ok_or_else(|| BanditError::BuilderError {
            message: "policy must be specified".to_string(),
        })?;

        // Validate policy parameters
        match &policy_type {
            LearningPolicy::EpsilonGreedy { epsilon } => {
                if !(0.0..=1.0).contains(epsilon) {
                    return Err(BanditError::InvalidParameter {
                        message: format!("epsilon must be between 0 and 1, got {}", epsilon),
                    });
                }
            }
            LearningPolicy::Ucb { alpha } => {
                if *alpha <= 0.0 {
                    return Err(BanditError::InvalidParameter {
                        message: format!("alpha must be positive, got {}", alpha),
                    });
                }
            }
            _ => {}
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

        // Create policy implementation
        // Note: We need Send + Sync + 'static here for the trait object
        let policy: Box<dyn Policy<A> + Send + Sync + 'static> = match &policy_type {
            LearningPolicy::Random => Box::new(Random),
            LearningPolicy::EpsilonGreedy { epsilon } => Box::new(EpsilonGreedy::new(*epsilon)),
            LearningPolicy::Ucb { .. } => {
                todo!("UCB implementation")
            }
            LearningPolicy::ThompsonSampling => {
                todo!("Thompson Sampling implementation")
            }
        };

        Ok(Bandit {
            arms,
            arm_set,
            arm_metadata,
            policy_type,
            policy,
        })
    }
}

impl<A> Default for BanditBuilder<A>
where
    A: Clone + Eq + Hash + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let bandit = Bandit::builder()
            .arms(vec!["a", "b", "c"])
            .policy(LearningPolicy::Random)
            .build()
            .unwrap();

        assert_eq!(bandit.n_arms(), 3);
        assert!(bandit.has_arm(&"a"));
        assert!(bandit.has_arm(&"b"));
        assert!(bandit.has_arm(&"c"));
        assert!(!bandit.has_arm(&"d"));
    }

    #[test]
    fn test_builder_with_policy() {
        let bandit = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
            .build()
            .unwrap();

        assert_eq!(bandit.n_arms(), 3);
    }

    #[test]
    fn test_builder_errors() {
        // No arms
        let result = Bandit::<i32>::builder()
            .policy(LearningPolicy::Random)
            .build();
        assert!(matches!(result, Err(BanditError::BuilderError { .. })));

        // No policy
        let result = Bandit::builder().arms(vec![1, 2, 3]).build();
        assert!(matches!(result, Err(BanditError::BuilderError { .. })));

        // Empty arms
        let result = Bandit::builder()
            .arms(Vec::<i32>::new())
            .policy(LearningPolicy::Random)
            .build();
        assert!(matches!(result, Err(BanditError::BuilderError { .. })));

        // Duplicate arms
        let result = Bandit::builder()
            .arms(vec![1, 2, 2, 3])
            .policy(LearningPolicy::Random)
            .build();
        assert!(matches!(result, Err(BanditError::BuilderError { .. })));

        // Invalid epsilon
        let result = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::EpsilonGreedy { epsilon: 1.5 })
            .build();
        assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));

        // Invalid alpha
        let result = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::Ucb { alpha: -1.0 })
            .build();
        assert!(matches!(result, Err(BanditError::InvalidParameter { .. })));
    }

    #[test]
    fn test_add_remove_arms() {
        let mut bandit = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::Random)
            .build()
            .unwrap();

        // Add new arm
        assert!(bandit.add_arm(4).is_ok());
        assert_eq!(bandit.n_arms(), 4);
        assert!(bandit.has_arm(&4));

        // Try to add existing arm
        assert!(matches!(
            bandit.add_arm(4),
            Err(BanditError::ArmAlreadyExists)
        ));

        // Remove arm
        assert!(bandit.remove_arm(&4).is_ok());
        assert_eq!(bandit.n_arms(), 3);
        assert!(!bandit.has_arm(&4));

        // Try to remove non-existent arm
        assert!(matches!(
            bandit.remove_arm(&4),
            Err(BanditError::ArmNotFound)
        ));
    }

    #[test]
    fn test_fit_validates_arms() {
        let mut bandit = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
            .build()
            .unwrap();

        // Valid fit
        assert!(bandit.fit(&[1, 2, 3], &[0.5, 0.8, 0.3]).is_ok());

        // Invalid arm
        assert!(matches!(
            bandit.fit(&[1, 2, 4], &[0.5, 0.8, 0.3]),
            Err(BanditError::ArmNotFound)
        ));

        // Mismatched lengths
        assert!(matches!(
            bandit.fit(&[1, 2], &[0.5, 0.8, 0.3]),
            Err(BanditError::DimensionMismatch { .. })
        ));
    }
}
