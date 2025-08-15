//! Arm type for multi-armed bandits.
//!
//! The `Arm` enum represents the different options/actions that can be selected
//! by a bandit algorithm. It supports integers, floats, and strings.

use ordered_float::OrderedFloat;
use std::fmt;
use std::hash::Hash;

/// Represents an arm (action/option) in a multi-armed bandit.
///
/// Arms can be created from integers, floats, or strings using the `From` trait.
///
/// # Examples
///
/// ```
/// use trashpanda::Arm;
///
/// let int_arm = Arm::from(42);
/// let float_arm = Arm::from(3.14);
/// let string_arm = Arm::from("option_a");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Arm {
    /// Integer arm
    Int(i64),
    /// Float arm (using OrderedFloat for Hash compatibility)
    Float(OrderedFloat<f64>),
    /// String arm
    String(String),
}

impl fmt::Display for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arm::Int(i) => write!(f, "{}", i),
            Arm::Float(fl) => write!(f, "{}", fl),
            Arm::String(s) => write!(f, "{}", s),
        }
    }
}

// From implementations for convenient arm creation

impl From<i64> for Arm {
    fn from(value: i64) -> Self {
        Arm::Int(value)
    }
}

impl From<i32> for Arm {
    fn from(value: i32) -> Self {
        Arm::Int(value as i64)
    }
}

impl From<usize> for Arm {
    fn from(value: usize) -> Self {
        Arm::Int(value as i64)
    }
}

impl From<f64> for Arm {
    fn from(value: f64) -> Self {
        Arm::Float(OrderedFloat(value))
    }
}

impl From<f32> for Arm {
    fn from(value: f32) -> Self {
        Arm::Float(OrderedFloat(value as f64))
    }
}

impl From<String> for Arm {
    fn from(value: String) -> Self {
        Arm::String(value)
    }
}

impl From<&str> for Arm {
    fn from(value: &str) -> Self {
        Arm::String(value.to_string())
    }
}

impl From<&String> for Arm {
    fn from(value: &String) -> Self {
        Arm::String(value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_creation() {
        // Test integer creation
        let arm1 = Arm::from(42i32);
        let arm2 = Arm::from(42i64);
        assert_eq!(arm1, arm2);

        // Test float creation - note f32 to f64 conversion may have precision differences
        let arm3 = Arm::from(3.14f64);
        let arm4 = Arm::from(3.14f64); // Use f64 for exact comparison
        assert_eq!(arm3, arm4);

        // Test that f32 conversion works (though values may differ slightly)
        let arm_f32 = Arm::from(3.14f32);
        assert!(matches!(arm_f32, Arm::Float(_)));

        // Test string creation
        let arm5 = Arm::from("test");
        let arm6 = Arm::from(String::from("test"));
        assert_eq!(arm5, arm6);
    }

    #[test]
    fn test_arm_equality() {
        let arm1 = Arm::from(1);
        let arm2 = Arm::from(1);
        let arm3 = Arm::from(2);

        assert_eq!(arm1, arm2);
        assert_ne!(arm1, arm3);
    }

    #[test]
    fn test_arm_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Arm::from(1));
        set.insert(Arm::from(3.14));
        set.insert(Arm::from("test"));

        assert!(set.contains(&Arm::from(1)));
        assert!(set.contains(&Arm::from(3.14)));
        assert!(set.contains(&Arm::from("test")));
        assert!(!set.contains(&Arm::from(2)));
    }

    #[test]
    fn test_arm_display() {
        assert_eq!(format!("{}", Arm::from(42)), "42");
        assert_eq!(format!("{}", Arm::from(3.14)), "3.14");
        assert_eq!(format!("{}", Arm::from("option")), "option");
    }
}
