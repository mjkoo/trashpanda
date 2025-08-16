//! Simple (non-contextual) bandit policies
//!
//! This module contains bandit algorithms that don't use contextual information.
//! These policies implement `Policy<A, ()>` where the context type is the unit type.

pub mod epsilon_greedy;
pub mod random;
pub mod thompson;
pub mod ucb;
