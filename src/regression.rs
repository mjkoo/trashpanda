use approx::abs_diff_ne;
use nalgebra::{DMatrix, DVector};

/// Ridge regression model for online learning
///
/// Implements ridge regression with Sherman-Morrison formula for
/// efficient online updates without full matrix inversions.
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    /// Regularization parameter (L2 penalty)
    pub l2_lambda: f64,
    /// Regression coefficients (beta)
    pub beta: DVector<f64>,
    /// A = X^T X + lambda * I
    pub a_matrix: DMatrix<f64>,
    /// A^-1 (inverse of A)
    pub a_inv: DMatrix<f64>,
    /// X^T y
    pub xty: DVector<f64>,
    /// Number of features
    pub num_features: usize,
}

impl RidgeRegression {
    /// Create a new ridge regression model
    pub fn new(num_features: usize, l2_lambda: f64) -> Self {
        let identity = DMatrix::<f64>::identity(num_features, num_features);
        let a_matrix = identity.clone() * l2_lambda;
        let a_inv = a_matrix.clone();
        let xty = DVector::<f64>::zeros(num_features);
        let beta = DVector::<f64>::zeros(num_features);

        Self {
            l2_lambda,
            beta,
            a_matrix,
            a_inv,
            xty,
            num_features,
        }
    }

    /// Fit the model with new data using online update
    ///
    /// Uses Sherman-Morrison formula for efficient inverse update
    pub fn fit(&mut self, x: &[f64], y: f64) {
        let x_vec = DVector::from_row_slice(x);

        // Update A = A + x * x^T
        self.a_matrix += &x_vec * x_vec.transpose();

        // Update A_inv using Sherman-Morrison formula
        // A_inv_new = A_inv - (A_inv * x * x^T * A_inv) / (1 + x^T * A_inv * x)
        let a_inv_x = &self.a_inv * &x_vec;
        let denominator = 1.0 + x_vec.dot(&a_inv_x);

        if abs_diff_ne!(denominator, 0.0, epsilon = 1e-10) {
            self.a_inv -= &a_inv_x * a_inv_x.transpose() / denominator;
        } else {
            // Fallback to direct inversion if numerical issues
            self.a_inv = self.a_matrix.clone().try_inverse().unwrap_or_else(|| {
                // If inversion fails, reset to identity
                DMatrix::<f64>::identity(self.num_features, self.num_features) * self.l2_lambda
            });
        }

        // Update X^T y
        self.xty += &x_vec * y;

        // Update beta = A_inv * X^T y
        self.beta = &self.a_inv * &self.xty;
    }

    /// Batch fit with multiple samples
    #[allow(dead_code)]
    pub fn fit_batch(&mut self, x: &[Vec<f64>], y: &[f64]) {
        for (xi, yi) in x.iter().zip(y.iter()) {
            self.fit(xi, *yi);
        }
    }

    /// Predict the output for a given input
    pub fn predict(&self, x: &[f64]) -> f64 {
        let x_vec = DVector::from_row_slice(x);
        x_vec.dot(&self.beta)
    }

    /// Get the variance (uncertainty) for a prediction
    ///
    /// Returns sqrt(x^T * A_inv * x) which is used for confidence bounds
    pub fn variance(&self, x: &[f64]) -> f64 {
        let x_vec = DVector::from_row_slice(x);
        let a_inv_x = &self.a_inv * &x_vec;
        x_vec.dot(&a_inv_x).sqrt()
    }

    /// Reset the model to initial state
    pub fn reset(&mut self) {
        let identity = DMatrix::<f64>::identity(self.num_features, self.num_features);
        self.a_matrix = identity.clone() * self.l2_lambda;
        self.a_inv = self.a_matrix.clone();
        self.xty = DVector::<f64>::zeros(self.num_features);
        self.beta = DVector::<f64>::zeros(self.num_features);
    }
}
