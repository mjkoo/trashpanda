//! Core bandit implementation and builder.

use crate::{Arm, BanditError, Result};
use std::collections::{HashMap, HashSet};

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
#[allow(dead_code)] // TODO: Remove when algorithms are implemented
pub struct Bandit {
    /// Available arms.
    arms: Vec<Arm>,
    /// Set of arms for O(1) membership checking.
    arm_set: HashSet<Arm>,
    /// Metadata for each arm.
    arm_metadata: HashMap<Arm, ArmMetadata>,
    /// Expected reward for each arm.
    arm_expectations: HashMap<Arm, f64>,
    /// Learning policy.
    policy: LearningPolicy,
    /// Random number generator.
    rng: rand::rngs::StdRng,
}

impl Bandit {
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
    pub fn builder() -> BanditBuilder {
        BanditBuilder::new()
    }

    /// Returns a slice of all arms in the bandit.
    pub fn arms(&self) -> &[Arm] {
        &self.arms
    }

    /// Returns the number of arms.
    pub fn n_arms(&self) -> usize {
        self.arms.len()
    }

    /// Checks if an arm exists in the bandit.
    pub fn has_arm(&self, arm: &Arm) -> bool {
        self.arm_set.contains(arm)
    }

    /// Adds a new arm to the bandit.
    ///
    /// # Errors
    ///
    /// Returns `BanditError::ArmAlreadyExists` if the arm already exists.
    pub fn add_arm(&mut self, arm: Arm) -> Result<()> {
        if self.has_arm(&arm) {
            return Err(BanditError::ArmAlreadyExists(arm));
        }

        self.arms.push(arm.clone());
        self.arm_set.insert(arm.clone());
        self.arm_metadata
            .insert(arm.clone(), ArmMetadata::default());
        self.arm_expectations.insert(arm, 0.0);

        Ok(())
    }

    /// Removes an arm from the bandit.
    ///
    /// # Errors
    ///
    /// Returns `BanditError::ArmNotFound` if the arm doesn't exist.
    pub fn remove_arm(&mut self, arm: &Arm) -> Result<()> {
        if !self.has_arm(arm) {
            return Err(BanditError::ArmNotFound(arm.clone()));
        }

        self.arms.retain(|a| a != arm);
        self.arm_set.remove(arm);
        self.arm_metadata.remove(arm);
        self.arm_expectations.remove(arm);

        Ok(())
    }

    /// Placeholder for fit implementation.
    pub fn fit(&mut self, _decisions: &[Arm], _rewards: &[f64]) -> Result<()> {
        // Will be implemented with specific policies
        todo!("Implement fit based on policy")
    }

    /// Placeholder for partial_fit implementation.
    pub fn partial_fit(&mut self, _decisions: &[Arm], _rewards: &[f64]) -> Result<()> {
        // Will be implemented with specific policies
        todo!("Implement partial_fit based on policy")
    }

    /// Placeholder for predict implementation.
    pub fn predict(&self) -> Result<Arm> {
        // Will be implemented with specific policies
        todo!("Implement predict based on policy")
    }

    /// Placeholder for predict_expectations implementation.
    pub fn predict_expectations(&self) -> Result<HashMap<Arm, f64>> {
        // Will be implemented with specific policies
        todo!("Implement predict_expectations based on policy")
    }
}

/// Builder for constructing a `Bandit`.
pub struct BanditBuilder {
    arms: Option<Vec<Arm>>,
    policy: Option<LearningPolicy>,
    seed: Option<u64>,
}

impl BanditBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            arms: None,
            policy: None,
            seed: None,
        }
    }

    /// Sets the arms for the bandit.
    ///
    /// # Examples
    ///
    /// ```
    /// use trashpanda::Bandit;
    ///
    /// let builder = Bandit::builder()
    ///     .arms(vec![1, 2, 3]);
    /// ```
    pub fn arms<I, A>(mut self, arms: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<Arm>,
    {
        self.arms = Some(arms.into_iter().map(|a| a.into()).collect());
        self
    }

    /// Sets the learning policy.
    pub fn policy(mut self, policy: LearningPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Sets the random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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
    pub fn build(self) -> Result<Bandit> {
        use rand::SeedableRng;

        let arms = self.arms.ok_or_else(|| BanditError::BuilderError {
            message: "arms must be specified".to_string(),
        })?;

        if arms.is_empty() {
            return Err(BanditError::BuilderError {
                message: "at least one arm must be specified".to_string(),
            });
        }

        let policy = self.policy.ok_or_else(|| BanditError::BuilderError {
            message: "policy must be specified".to_string(),
        })?;

        // Validate policy parameters
        match &policy {
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

        // Initialize metadata and expectations
        let mut arm_metadata = HashMap::new();
        let mut arm_expectations = HashMap::new();
        for arm in &arms {
            arm_metadata.insert(arm.clone(), ArmMetadata::default());
            arm_expectations.insert(arm.clone(), 0.0);
        }

        // Create RNG
        let rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        Ok(Bandit {
            arms,
            arm_set,
            arm_metadata,
            arm_expectations,
            policy,
            rng,
        })
    }
}

impl Default for BanditBuilder {
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
        assert!(bandit.has_arm(&Arm::from("a")));
        assert!(bandit.has_arm(&Arm::from("b")));
        assert!(bandit.has_arm(&Arm::from("c")));
        assert!(!bandit.has_arm(&Arm::from("d")));
    }

    #[test]
    fn test_builder_with_seed() {
        let bandit = Bandit::builder()
            .arms(vec![1, 2, 3])
            .policy(LearningPolicy::EpsilonGreedy { epsilon: 0.1 })
            .seed(42)
            .build()
            .unwrap();

        assert_eq!(bandit.n_arms(), 3);
    }

    #[test]
    fn test_builder_errors() {
        // No arms
        let result = Bandit::builder().policy(LearningPolicy::Random).build();
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
        assert!(bandit.add_arm(Arm::from(4)).is_ok());
        assert_eq!(bandit.n_arms(), 4);
        assert!(bandit.has_arm(&Arm::from(4)));

        // Try to add existing arm
        assert!(matches!(
            bandit.add_arm(Arm::from(4)),
            Err(BanditError::ArmAlreadyExists(_))
        ));

        // Remove arm
        assert!(bandit.remove_arm(&Arm::from(4)).is_ok());
        assert_eq!(bandit.n_arms(), 3);
        assert!(!bandit.has_arm(&Arm::from(4)));

        // Try to remove non-existent arm
        assert!(matches!(
            bandit.remove_arm(&Arm::from(4)),
            Err(BanditError::ArmNotFound(_))
        ));
    }
}
