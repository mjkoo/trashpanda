use faer::prelude::*;
use faer::{Mat, MatRef};
use linfa::prelude::*;
use linfa_elasticnet::{ElasticNet, ElasticNetParams};
use ndarray::{Array1, Array2, Axis};

/// Ridge regression model for online learning
///
/// Uses ElasticNet with L1 penalty set to 0, effectively making it Ridge Regression.
/// Maintains training data for retraining when needed. Uses faer for high-performance
/// linear algebra operations.
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    /// Regularization parameter (L2 penalty)
    pub l2_lambda: f64,
    /// Number of features
    pub num_features: usize,
    /// Fitted model (if trained)
    model: Option<ElasticNet<f64>>,
    /// Training data X
    x_data: Vec<Array1<f64>>,
    /// Training data y
    y_data: Vec<f64>,
    /// Cached coefficients for fast access
    pub beta: Array1<f64>,
    /// Cached A matrix for variance computation (X^T X + lambda * I)
    pub a_matrix: Array2<f64>,
    /// Cached inverse of A matrix
    pub a_inv: Array2<f64>,
    /// Cached X^T y for internal use
    pub xty: Array1<f64>,
}

impl RidgeRegression {
    /// Create a new ridge regression model
    pub fn new(num_features: usize, l2_lambda: f64) -> Self {
        let mut identity = Array2::<f64>::zeros((num_features, num_features));
        identity.diag_mut().fill(1.0);
        let a_matrix = &identity * l2_lambda;
        let a_inv = a_matrix.clone();

        Self {
            l2_lambda,
            num_features,
            model: None,
            x_data: Vec::new(),
            y_data: Vec::new(),
            beta: Array1::<f64>::zeros(num_features),
            a_matrix,
            a_inv,
            xty: Array1::<f64>::zeros(num_features),
        }
    }

    /// Fit the model with new data using online update
    ///
    /// Accumulates data and uses incremental matrix updates for efficiency
    pub fn fit(&mut self, x: &[f64], y: f64) {
        let x_vec = Array1::from(x.to_vec());

        // Store the data point
        self.x_data.push(x_vec.clone());
        self.y_data.push(y);

        // Update A = A + x * x^T (for variance computation)
        let x_outer = x_vec.view().insert_axis(Axis(1));
        let x_outer_t = x_vec.view().insert_axis(Axis(0));
        self.a_matrix += &x_outer.dot(&x_outer_t);

        // Update A_inv using Sherman-Morrison formula
        let a_inv_x = self.a_inv.dot(&x_vec);
        let denominator = 1.0 + x_vec.dot(&a_inv_x);

        if denominator.abs() > 1e-10 {
            let update = a_inv_x.view().insert_axis(Axis(1));
            let update_t = a_inv_x.view().insert_axis(Axis(0));
            self.a_inv -= &(update.dot(&update_t) / denominator);
        } else {
            // Fallback to direct inversion using faer
            self.a_inv = self.invert_matrix_faer(&self.a_matrix);
        }

        // Update X^T y
        self.xty = &self.xty + &(&x_vec * y);

        // Update beta = A_inv * X^T y (closed-form solution)
        self.beta = self.a_inv.dot(&self.xty);

        // Retrain ElasticNet model periodically or when needed
        // We'll retrain every 10 samples or when explicitly needed
        if self.x_data.len() % 10 == 0 || self.model.is_none() {
            self.retrain();
        }
    }

    /// Invert a matrix using faer's high-performance implementation
    fn invert_matrix_faer(&self, matrix: &Array2<f64>) -> Array2<f64> {
        // Convert ndarray to faer matrix
        let n = self.num_features;
        let mut faer_mat = Mat::<f64>::zeros(n, n);

        // Copy data from ndarray to faer
        for i in 0..n {
            for j in 0..n {
                faer_mat[(i, j)] = matrix[[i, j]];
            }
        }

        // Compute inverse using faer's LU decomposition
        let lu = faer_mat.partial_piv_lu();
        let mut inv = Mat::<f64>::zeros(n, n);

        // Solve for each column of the identity matrix to get the inverse
        for i in 0..n {
            let mut e = Mat::<f64>::zeros(n, 1);
            e[(i, 0)] = 1.0;
            let sol = lu.solve(&e);
            for j in 0..n {
                inv[(j, i)] = sol[(j, 0)];
            }
        }
        let inverse_faer = inv;

        // Convert back to ndarray manually
        let mut result = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = inverse_faer[(i, j)];
            }
        }
        result
    }

    /// Retrain the ElasticNet model with all accumulated data
    fn retrain(&mut self) {
        if self.x_data.is_empty() {
            return;
        }

        // Convert data to the format expected by linfa
        let n_samples = self.x_data.len();
        let mut x_matrix = Array2::<f64>::zeros((n_samples, self.num_features));
        for (i, x) in self.x_data.iter().enumerate() {
            x_matrix.row_mut(i).assign(x);
        }

        let y_array = Array1::from(self.y_data.clone());

        // Create dataset
        let dataset = Dataset::new(x_matrix, y_array);

        // Configure ElasticNet as Ridge (L1 ratio = 0)
        let params = ElasticNetParams::default()
            .penalty(self.l2_lambda)
            .l1_ratio(0.0) // Pure Ridge regression
            .with_intercept(false) // No intercept for compatibility
            .max_iterations(1000)
            .tolerance(1e-4);

        // Validate parameters and fit the model
        if let Ok(valid_params) = params.check() {
            match valid_params.fit(&dataset) {
                Ok(fitted_model) => {
                    self.model = Some(fitted_model);
                    // Note: ElasticNet coefficients might differ slightly from our
                    // closed-form solution due to optimization vs analytical solution
                }
                Err(_) => {
                    // If fitting fails, we'll continue using our closed-form solution
                }
            }
        }
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
        // Use our closed-form solution (more stable for online learning)
        let x_vec = Array1::from(x.to_vec());
        x_vec.dot(&self.beta)
    }

    /// Get the variance (uncertainty) for a prediction
    ///
    /// Returns sqrt(x^T * A_inv * x) which is used for confidence bounds
    pub fn variance(&self, x: &[f64]) -> f64 {
        let x_vec = Array1::from(x.to_vec());
        let a_inv_x = self.a_inv.dot(&x_vec);
        x_vec.dot(&a_inv_x).sqrt()
    }

    /// Reset the model to initial state
    pub fn reset(&mut self) {
        self.x_data.clear();
        self.y_data.clear();
        self.model = None;

        let mut identity = Array2::<f64>::zeros((self.num_features, self.num_features));
        identity.diag_mut().fill(1.0);
        self.a_matrix = &identity * self.l2_lambda;
        self.a_inv = self.a_matrix.clone();
        self.xty = Array1::<f64>::zeros(self.num_features);
        self.beta = Array1::<f64>::zeros(self.num_features);
    }
}
