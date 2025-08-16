use approx::abs_diff_ne;
use faer::{Mat, linalg::solvers::DenseSolveCore};

/// Ridge regression model for online learning
///
/// Implements ridge regression with Sherman-Morrison formula for
/// efficient online updates without full matrix inversions.
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    /// Regularization parameter (L2 penalty)
    pub l2_lambda: f64,
    /// Regression coefficients (beta)
    pub beta: Mat<f64>,
    /// A = X^T X + lambda * I
    pub a_matrix: Mat<f64>,
    /// A^-1 (inverse of A)
    pub a_inv: Mat<f64>,
    /// X^T y
    pub xty: Mat<f64>,
    /// Number of features
    pub num_features: usize,
}

impl RidgeRegression {
    /// Create a new ridge regression model
    pub fn new(num_features: usize, l2_lambda: f64) -> Self {
        let mut a_matrix = Mat::<f64>::zeros(num_features, num_features);

        // Set diagonal to l2_lambda (lambda * I)
        for i in 0..num_features {
            a_matrix[(i, i)] = l2_lambda;
        }

        let a_inv = a_matrix.clone();
        let xty = Mat::<f64>::zeros(num_features, 1);
        let beta = Mat::<f64>::zeros(num_features, 1);

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
        // Convert x to column vector
        let x_vec = Mat::from_fn(self.num_features, 1, |i, _| x[i]);

        // Update A = A + x * x^T
        // A_new = A + x * x^T
        let mut xx_transpose = Mat::<f64>::zeros(self.num_features, self.num_features);
        for i in 0..self.num_features {
            for j in 0..self.num_features {
                xx_transpose[(i, j)] = x[i] * x[j];
            }
        }

        for i in 0..self.num_features {
            for j in 0..self.num_features {
                self.a_matrix[(i, j)] += xx_transpose[(i, j)];
            }
        }

        // Update A_inv using Sherman-Morrison formula
        // A_inv_new = A_inv - (A_inv * x * x^T * A_inv) / (1 + x^T * A_inv * x)
        let a_inv_x = &self.a_inv * &x_vec;

        // Calculate x^T * A_inv * x (scalar)
        let mut denominator = 1.0;
        for i in 0..self.num_features {
            denominator += x[i] * a_inv_x[(i, 0)];
        }

        if abs_diff_ne!(denominator, 0.0, epsilon = 1e-10) {
            // Calculate A_inv * x * x^T * A_inv / denominator and subtract from A_inv
            for i in 0..self.num_features {
                for j in 0..self.num_features {
                    let update = (a_inv_x[(i, 0)] * a_inv_x[(j, 0)]) / denominator;
                    self.a_inv[(i, j)] -= update;
                }
            }
        } else {
            // Fallback to direct inversion if numerical issues
            let lu = self.a_matrix.partial_piv_lu();
            // Note: In faer, inverse() returns a Mat directly, not a Result
            self.a_inv = lu.inverse();
        }

        // Update X^T y
        for (i, &xi) in x.iter().enumerate().take(self.num_features) {
            self.xty[(i, 0)] += xi * y;
        }

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
        let mut result = 0.0;
        for (i, &xi) in x.iter().enumerate().take(self.num_features) {
            result += xi * self.beta[(i, 0)];
        }
        result
    }

    /// Get the variance (uncertainty) for a prediction
    ///
    /// Returns sqrt(x^T * A_inv * x) which is used for confidence bounds
    pub fn variance(&self, x: &[f64]) -> f64 {
        // Calculate A_inv * x
        let mut a_inv_x = vec![0.0; self.num_features];
        for (i, a_inv_x_i) in a_inv_x.iter_mut().enumerate().take(self.num_features) {
            for (j, &xj) in x.iter().enumerate().take(self.num_features) {
                *a_inv_x_i += self.a_inv[(i, j)] * xj;
            }
        }

        // Calculate x^T * A_inv * x
        let mut result = 0.0;
        for (i, &xi) in x.iter().enumerate().take(self.num_features) {
            result += xi * a_inv_x[i];
        }

        result.sqrt()
    }

    /// Reset the model to initial state
    pub fn reset(&mut self) {
        // Reset a_matrix to lambda * I
        self.a_matrix = Mat::<f64>::zeros(self.num_features, self.num_features);
        for i in 0..self.num_features {
            self.a_matrix[(i, i)] = self.l2_lambda;
        }

        self.a_inv = self.a_matrix.clone();
        self.xty = Mat::<f64>::zeros(self.num_features, 1);
        self.beta = Mat::<f64>::zeros(self.num_features, 1);
    }
}
