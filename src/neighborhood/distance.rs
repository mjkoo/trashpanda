//! Distance metrics for neighborhood-based algorithms

use std::f64;

/// Trait for distance metrics between context vectors
pub trait DistanceMetric: Clone {
    /// Calculate distance between two context vectors
    fn distance(&self, a: &[f64], b: &[f64]) -> f64;
}

/// Euclidean distance metric
#[derive(Clone, Debug, Default)]
pub struct Euclidean;

impl DistanceMetric for Euclidean {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Manhattan (L1) distance metric
#[derive(Clone, Debug, Default)]
pub struct Manhattan;

impl DistanceMetric for Manhattan {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }
}

/// Cosine distance metric (1 - cosine similarity)
#[derive(Clone, Debug, Default)]
pub struct Cosine;

impl DistanceMetric for Cosine {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");

        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0; // Maximum distance if one vector is zero
        }

        1.0 - (dot_product / (norm_a * norm_b))
    }
}

/// Chebyshev (L-infinity) distance metric
#[derive(Clone, Debug, Default)]
pub struct Chebyshev;

impl DistanceMetric for Chebyshev {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }
}

/// Minkowski distance metric with parameter p
#[derive(Clone, Debug)]
pub struct Minkowski {
    p: f64,
}

impl Minkowski {
    /// Create a new Minkowski distance metric with parameter p
    pub fn new(p: f64) -> Self {
        assert!(p >= 1.0, "Minkowski parameter p must be >= 1.0");
        Self { p }
    }
}

impl DistanceMetric for Minkowski {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        if self.p == 1.0 {
            // Manhattan distance
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        } else if self.p == 2.0 {
            // Euclidean distance
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        } else if self.p.is_infinite() {
            // Chebyshev distance
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs().powf(self.p))
                .sum::<f64>()
                .powf(1.0 / self.p)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_distance() {
        let metric = Euclidean;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_relative_eq!(metric.distance(&a, &b), 5.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let metric = Manhattan;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_relative_eq!(metric.distance(&a, &b), 7.0);
    }

    #[test]
    fn test_cosine_distance() {
        let metric = Cosine;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_relative_eq!(metric.distance(&a, &b), 1.0); // Orthogonal vectors

        let c = vec![1.0, 1.0];
        let d = vec![2.0, 2.0];
        assert_relative_eq!(metric.distance(&c, &d), 0.0, epsilon = 1e-10); // Parallel vectors
    }

    #[test]
    fn test_chebyshev_distance() {
        let metric = Chebyshev;
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_relative_eq!(metric.distance(&a, &b), 4.0);
    }

    #[test]
    fn test_minkowski_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        // p=1 should be Manhattan
        let metric1 = Minkowski::new(1.0);
        assert_relative_eq!(metric1.distance(&a, &b), 7.0);

        // p=2 should be Euclidean
        let metric2 = Minkowski::new(2.0);
        assert_relative_eq!(metric2.distance(&a, &b), 5.0);

        // p=3
        let metric3 = Minkowski::new(3.0);
        let expected = (3.0_f64.powi(3) + 4.0_f64.powi(3)).powf(1.0 / 3.0);
        assert_relative_eq!(metric3.distance(&a, &b), expected);
    }
}
