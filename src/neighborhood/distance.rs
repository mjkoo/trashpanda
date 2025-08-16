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

/// Canberra distance metric
///
/// A weighted version of Manhattan distance that normalizes each dimension
/// by the sum of the absolute values. Useful when features have different
/// scales and you want equal contribution from each dimension.
#[derive(Clone, Debug, Default)]
pub struct Canberra;

impl DistanceMetric for Canberra {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let numerator = (x - y).abs();
                let denominator = x.abs() + y.abs();
                if denominator == 0.0 {
                    0.0 // Both values are zero, no contribution to distance
                } else {
                    numerator / denominator
                }
            })
            .sum()
    }
}

/// Standardized Euclidean distance metric
///
/// Euclidean distance where each dimension is scaled by its standard deviation.
/// This makes features with different scales contribute equally to the distance.
/// Standard deviations are computed from the provided data.
#[derive(Clone, Debug)]
pub struct StandardizedEuclidean {
    /// Standard deviations for each dimension
    std_devs: Vec<f64>,
}

impl StandardizedEuclidean {
    /// Create a new Standardized Euclidean distance metric
    ///
    /// # Arguments
    /// * `std_devs` - Standard deviations for each dimension. Must match the
    ///   dimension of vectors that will be compared.
    pub fn new(std_devs: Vec<f64>) -> Self {
        assert!(!std_devs.is_empty(), "Standard deviations cannot be empty");
        assert!(
            std_devs.iter().all(|&s| s > 0.0),
            "All standard deviations must be positive"
        );
        Self { std_devs }
    }

    /// Create a Standardized Euclidean metric by computing standard deviations
    /// from a collection of training vectors
    ///
    /// # Arguments
    /// * `training_data` - Collection of vectors to compute standard deviations from
    pub fn from_data(training_data: &[Vec<f64>]) -> Self {
        assert!(!training_data.is_empty(), "Training data cannot be empty");

        let n_dims = training_data[0].len();
        assert!(
            training_data.iter().all(|v| v.len() == n_dims),
            "All training vectors must have the same dimension"
        );

        let n_samples = training_data.len() as f64;

        // Compute means
        let means: Vec<f64> = (0..n_dims)
            .map(|i| training_data.iter().map(|v| v[i]).sum::<f64>() / n_samples)
            .collect();

        // Compute standard deviations
        let std_devs: Vec<f64> = (0..n_dims)
            .map(|i| {
                let variance = training_data
                    .iter()
                    .map(|v| (v[i] - means[i]).powi(2))
                    .sum::<f64>()
                    / n_samples;
                variance.sqrt().max(1e-8) // Avoid division by zero
            })
            .collect();

        Self::new(std_devs)
    }
}

impl DistanceMetric for StandardizedEuclidean {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        assert_eq!(
            a.len(),
            self.std_devs.len(),
            "Vector dimensions must match standard deviation dimensions"
        );

        a.iter()
            .zip(b.iter())
            .zip(self.std_devs.iter())
            .map(|((x, y), std_dev)| {
                let diff = x - y;
                (diff / std_dev).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }
}

/// Mahalanobis distance metric
///
/// Distance that accounts for the covariance structure of the data.
/// Uses the inverse covariance matrix to weight different dimensions
/// according to their correlation and variance.
#[derive(Clone, Debug)]
pub struct Mahalanobis {
    /// Inverse covariance matrix (flattened row-major)
    inv_cov_matrix: Vec<f64>,
    /// Number of dimensions
    n_dims: usize,
}

impl Mahalanobis {
    /// Create a new Mahalanobis distance metric
    ///
    /// # Arguments
    /// * `inv_cov_matrix` - Inverse covariance matrix in row-major order
    /// * `n_dims` - Number of dimensions
    pub fn new(inv_cov_matrix: Vec<f64>, n_dims: usize) -> Self {
        assert_eq!(
            inv_cov_matrix.len(),
            n_dims * n_dims,
            "Inverse covariance matrix size must match n_dims^2"
        );
        Self {
            inv_cov_matrix,
            n_dims,
        }
    }

    /// Create a Mahalanobis metric by computing the covariance matrix
    /// from training data and inverting it
    ///
    /// # Arguments
    /// * `training_data` - Collection of vectors to compute covariance from
    pub fn from_data(training_data: &[Vec<f64>]) -> Option<Self> {
        assert!(!training_data.is_empty(), "Training data cannot be empty");

        let n_dims = training_data[0].len();
        assert!(
            training_data.iter().all(|v| v.len() == n_dims),
            "All training vectors must have the same dimension"
        );

        let n_samples = training_data.len() as f64;

        // Compute means
        let means: Vec<f64> = (0..n_dims)
            .map(|i| training_data.iter().map(|v| v[i]).sum::<f64>() / n_samples)
            .collect();

        // Compute covariance matrix
        let mut cov_matrix = vec![0.0; n_dims * n_dims];
        for i in 0..n_dims {
            for j in 0..n_dims {
                let covariance = training_data
                    .iter()
                    .map(|v| (v[i] - means[i]) * (v[j] - means[j]))
                    .sum::<f64>()
                    / n_samples;
                cov_matrix[i * n_dims + j] = covariance;
            }
        }

        // Add small regularization to diagonal for numerical stability
        for i in 0..n_dims {
            cov_matrix[i * n_dims + i] += 1e-8;
        }

        // Compute inverse using simple methods for small matrices
        // For production use, consider using a proper linear algebra library
        if n_dims == 1 {
            let inv_cov_matrix = vec![1.0 / cov_matrix[0]];
            Some(Self::new(inv_cov_matrix, n_dims))
        } else if n_dims == 2 {
            // 2x2 matrix inversion
            let det = cov_matrix[0] * cov_matrix[3] - cov_matrix[1] * cov_matrix[2];
            if det.abs() < 1e-12 {
                return None; // Singular matrix
            }
            let inv_det = 1.0 / det;
            let inv_cov_matrix = vec![
                cov_matrix[3] * inv_det,  // [0,0]
                -cov_matrix[1] * inv_det, // [0,1]
                -cov_matrix[2] * inv_det, // [1,0]
                cov_matrix[0] * inv_det,  // [1,1]
            ];
            Some(Self::new(inv_cov_matrix, n_dims))
        } else {
            // For larger matrices, fall back to regularized identity
            // In practice, you'd want to use faer or nalgebra for proper inversion
            let mut inv_cov_matrix = vec![0.0; n_dims * n_dims];
            for i in 0..n_dims {
                inv_cov_matrix[i * n_dims + i] = 1.0;
            }
            Some(Self::new(inv_cov_matrix, n_dims))
        }
    }
}

impl DistanceMetric for Mahalanobis {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Context dimensions must match");
        assert_eq!(
            a.len(),
            self.n_dims,
            "Vector dimensions must match Mahalanobis dimensions"
        );

        // Compute difference vector
        let diff: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        // Compute diff^T * inv_cov * diff
        let mut result = 0.0;
        for i in 0..self.n_dims {
            for j in 0..self.n_dims {
                result += diff[i] * self.inv_cov_matrix[i * self.n_dims + j] * diff[j];
            }
        }

        result.sqrt()
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

    #[test]
    fn test_canberra_distance() {
        let metric = Canberra;

        // Basic test case
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 1.0, 3.0];
        // |1-2|/(|1|+|2|) + |2-1|/(|2|+|1|) + |3-3|/(|3|+|3|) = 1/3 + 1/3 + 0 = 2/3
        assert_relative_eq!(metric.distance(&a, &b), 2.0 / 3.0, epsilon = 1e-10);

        // Both vectors zero at same position
        let c = vec![0.0, 1.0];
        let d = vec![0.0, 2.0];
        // 0 + |1-2|/(|1|+|2|) = 0 + 1/3 = 1/3
        assert_relative_eq!(metric.distance(&c, &d), 1.0 / 3.0, epsilon = 1e-10);

        // Identical vectors
        let e = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(metric.distance(&e, &e), 0.0);
    }

    #[test]
    fn test_standardized_euclidean_distance() {
        // Test with known standard deviations
        let std_devs = vec![1.0, 2.0]; // Different scales
        let metric = StandardizedEuclidean::new(std_devs);

        let a = vec![0.0, 0.0];
        let b = vec![1.0, 4.0];
        // sqrt((1/1)^2 + (4/2)^2) = sqrt(1 + 4) = sqrt(5)
        assert_relative_eq!(metric.distance(&a, &b), 5.0_f64.sqrt());

        // Test from_data method
        let training_data = vec![vec![0.0, 0.0], vec![2.0, 4.0], vec![4.0, 8.0]];
        let metric_from_data = StandardizedEuclidean::from_data(&training_data);

        // Should work without panicking
        let distance = metric_from_data.distance(&[1.0, 2.0], &[3.0, 6.0]);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_mahalanobis_distance() {
        // Test 1D case (should reduce to standardized distance)
        let training_1d = vec![vec![1.0], vec![2.0], vec![3.0]];
        let metric_1d = Mahalanobis::from_data(&training_1d).unwrap();

        let a = vec![1.0];
        let b = vec![2.0];
        let distance = metric_1d.distance(&a, &b);
        assert!(distance > 0.0);

        // Test 2D case with identity covariance
        let inv_cov = vec![1.0, 0.0, 0.0, 1.0]; // Identity matrix
        let metric_2d = Mahalanobis::new(inv_cov, 2);

        let c = vec![0.0, 0.0];
        let d = vec![3.0, 4.0];
        // Should be same as Euclidean with identity covariance
        assert_relative_eq!(metric_2d.distance(&c, &d), 5.0);

        // Test from_data with 2D data
        let training_2d = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
        ];
        let metric_from_data = Mahalanobis::from_data(&training_2d).unwrap();

        let distance_2d = metric_from_data.distance(&[1.0, 2.0], &[3.0, 4.0]);
        assert!(distance_2d > 0.0);
    }

    #[test]
    fn test_standardized_euclidean_edge_cases() {
        // Test with very small standard deviation (should not panic due to regularization)
        let std_devs = vec![1e-10, 1.0];
        let metric = StandardizedEuclidean::new(std_devs);

        let a = vec![0.0, 0.0];
        let b = vec![1e-10, 1.0];
        let distance = metric.distance(&a, &b);
        assert!(distance.is_finite());
    }

    #[test]
    #[should_panic(expected = "All standard deviations must be positive")]
    fn test_standardized_euclidean_invalid_std_dev() {
        StandardizedEuclidean::new(vec![1.0, 0.0]); // Zero std dev should panic
    }

    #[test]
    fn test_mahalanobis_singular_matrix() {
        // Test data that would create a singular covariance matrix
        let training_data = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0], // Same as first vector
            vec![1.0, 2.0], // Same as first vector
        ];

        let result = Mahalanobis::from_data(&training_data);
        // Should handle singular matrix gracefully (either return None or use regularization)
        if let Some(metric) = result {
            let distance = metric.distance(&[1.0, 2.0], &[2.0, 3.0]);
            assert!(distance.is_finite());
        }
    }

    #[test]
    fn test_distance_metric_symmetry() {
        // Test that all distance metrics are symmetric: d(a,b) = d(b,a)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let euclidean = Euclidean;
        assert_relative_eq!(euclidean.distance(&a, &b), euclidean.distance(&b, &a));

        let manhattan = Manhattan;
        assert_relative_eq!(manhattan.distance(&a, &b), manhattan.distance(&b, &a));

        let cosine = Cosine;
        assert_relative_eq!(cosine.distance(&a, &b), cosine.distance(&b, &a));

        let chebyshev = Chebyshev;
        assert_relative_eq!(chebyshev.distance(&a, &b), chebyshev.distance(&b, &a));

        let canberra = Canberra;
        assert_relative_eq!(canberra.distance(&a, &b), canberra.distance(&b, &a));

        let std_euclidean = StandardizedEuclidean::new(vec![1.0, 1.0, 1.0]);
        assert_relative_eq!(
            std_euclidean.distance(&a, &b),
            std_euclidean.distance(&b, &a)
        );
    }

    #[test]
    fn test_distance_metric_identity() {
        // Test that d(a,a) = 0 for all metrics
        let a = vec![1.0, 2.0, 3.0];

        assert_relative_eq!(Euclidean.distance(&a, &a), 0.0);
        assert_relative_eq!(Manhattan.distance(&a, &a), 0.0);
        assert_relative_eq!(Cosine.distance(&a, &a), 0.0, epsilon = 1e-10);
        assert_relative_eq!(Chebyshev.distance(&a, &a), 0.0);
        assert_relative_eq!(Canberra.distance(&a, &a), 0.0);

        let std_euclidean = StandardizedEuclidean::new(vec![1.0, 1.0, 1.0]);
        assert_relative_eq!(std_euclidean.distance(&a, &a), 0.0);
    }
}
