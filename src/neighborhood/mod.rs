//! Neighborhood-based bandit policies
//!
//! This module contains bandit algorithms that use similarity-based approaches,
//! finding nearest neighbors in the historical data to make decisions.

pub mod clusters;
pub mod distance;
pub mod knearest;
pub mod radius;
