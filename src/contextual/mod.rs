//! Contextual bandit policies
//!
//! This module contains bandit algorithms that use contextual information to make decisions.
//! These policies implement `Policy<A, &[f64]>` where the context is a feature vector.

pub mod lingreedy;
pub mod lints;
pub mod linucb;
