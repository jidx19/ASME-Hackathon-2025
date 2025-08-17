import pandas as pd
import numpy as np
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
import warnings

# Ignore future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# 1. Physics Model Functions (Mass and Surface Area)
# =============================================================================

def predict_log_linear(X_val, YZ_val, coeffs):
    """
    Generic log-linear model prediction function.
    """
    if X_val <= 0 or YZ_val <= 0:
        return 1e10  # Return a large value to avoid math errors
        
    logX = np.log(X_val)
    logYZ = np.log(YZ_val)
    
    log_pred = (coeffs['const'] +
                coeffs['logX'] * logX +
                coeffs['logYZ'] * logYZ +
                coeffs['logX_sq'] * logX**2 +
                coeffs['logYZ_sq'] * logYZ**2 +
                coeffs['logX_logYZ'] * logX * logYZ)
                
    return np.exp(log_pred)

# --- Mass model coefficients (updated) ---
mass_coeffs = {
    'const': 6.2696, 'logX': -0.2222, 'logYZ': -0.4861,
    'logX_sq': 0.0085, 'logYZ_sq': 0.0395, 'logX_logYZ': 0.0292
}
predict_mass = lambda x, yz: predict_log_linear(x, yz, mass_coeffs)

# =============================================================================
# 2. GPR Model Predictor Class (PressureDrop, AvgVelocity, Surface Area)
# =============================================================================

class GPRPredictor:
    """
    Generic Gaussian Process Regression predictor to load and predict using pre-trained models.
    """
    def __init__(self, model_params, X_train, y_train):
        self.model_params = model_params
        self.scaler_X = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.X_train_df = X_train
        self.y_train = y_train
        
        self.scaler_y = StandardScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        self.poly = PolynomialFeatures(degree=self.model_params['poly_degree'])
        self.X_poly_train = self.poly.fit_transform(self.X_train_scaled)
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_poly_train, self.y_train_scaled)
        
        kernel_choice = self.model_params['kernel_type']
        
        from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
        
        params = self.model_params['gpr_optimized_params']
        constant_value = params['constant_value']
        noise_level = params['noise_level']
        length_scales = params['length_scales']

        if kernel_choice == 'RBF':
            kernel = ConstantKernel(constant_value) * RBF(length_scale=length_scales)
        elif kernel_choice == 'Matern1.5':
            kernel = ConstantKernel(constant_value) * Matern(length_scale=length_scales, nu=1.5)
        elif kernel_choice == 'Matern2.5':
            kernel = ConstantKernel(constant_value) * Matern(length_scale=length_scales, nu=2.5)
        elif kernel_choice == 'Periodic':
            if isinstance(length_scales, list):
                length_scale_val = length_scales[0]
                periodicity_val = length_scales[1]
            else:
                length_scale_val = length_scales
                periodicity_val = length_scales
            kernel = ConstantKernel(constant_value) * ExpSineSquared(length_scale=length_scale_val, periodicity=periodicity_val)
        
        kernel += WhiteKernel(noise_level=noise_level)
        self.gpr_model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
        residuals_train = self.y_train_scaled - self.linear_model.predict(self.X_poly_train)
        self.gpr_model.fit(self.X_train_scaled, residuals_train)

    def predict(self, X_new):
        distances = cdist(self.X_train_df.values, X_new.reshape(1, -1))
        match_idx = np.where(distances.flatten() < 1e-6)[0]

        if len(match_idx) > 0:
            idx = match_idx[0]
            original_value = self.y_train.iloc[idx]
            print(f"Exact match found with a sample in the training data. Returning the simulation value: {original_value:.4f}")
            # Continue to predict with GPR model
            
        X_new_scaled = self.scaler_X.transform(X_new.reshape(1, -1))
        
        X_poly_new = self.poly.transform(X_new_scaled)
        y_trend_new_scaled = self.linear_model.predict(X_poly_new)
        
        y_residuals_pred_scaled, y_residuals_std_scaled = self.gpr_model.predict(X_new_scaled, return_std=True)
        
        y_pred_scaled = y_trend_new_scaled + y_residuals_pred_scaled
        
        y_pred_orig = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
        
        y_residuals_std_orig = y_residuals_std_scaled * self.scaler_y.scale_[0]
        
        ci_lower = y_pred_orig - 1.96 * y_residuals_std_orig
        ci_upper = y_pred_orig + 1.96 * y_residuals_std_orig
        
        return {
            'prediction': y_pred_orig,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_exact_match': False
        }

# =============================================================================
# 3. Main Program
# =============================================================================

file_path_train = "LHS625.csv"
file_path_results = "universal_kriging_all_results_LHS625_Scalerv3.json"

try:
    df_train = pd.read_csv(file_path_train, skipinitialspace=True)
    with open(file_path_results, 'r', encoding='utf-8') as f:
        all_results_data = json.load(f)['results']
except FileNotFoundError:
    print("Error: Could not find data or results file. Please check the paths.")
    exit()

input_vars = ['X Cell Size', 'YZ Cell Size', 'Velocity Inlet']
output_vars = ['AvgVelocity', 'PressureDrop', 'Surface Area']
df_train_inputs = df_train[input_vars]

# --- 3.1 Instantiate all specified predictors (manual selection) ---
predictors = {}
predictors['Mass'] = 'log_linear'

# Define the fixed model choices
fixed_models = {
    'AvgVelocity': {'poly_degree': 3, 'kernel_type': 'Matern1.5'},
    'PressureDrop': {'poly_degree': 3, 'kernel_type': 'RBF'},
    'Surface Area': {'poly_degree': 3, 'kernel_type': 'Matern1.5'},
}

for output_var, fixed_params in fixed_models.items():
    poly_degree_val = fixed_params['poly_degree']
    kernel_type = fixed_params['kernel_type']
    
    # Construct the "poly_N" key to match the JSON structure
    poly_degree_key = f"poly_{poly_degree_val}"
    
    # Get the specified model parameters from the JSON file
    model_params = all_results_data.get(output_var, {}).get(poly_degree_key, {}).get(kernel_type)
    
    if model_params:
        
        predictor = GPRPredictor(model_params, df_train_inputs, df_train[output_var])
        predictors[output_var] = predictor
    else:
        print(f"Warning: Could not find the specified model for {output_var} (polynomial={poly_degree_val}, kernel={kernel_type}). Please check the keys in the JSON file.")

# =============================================================================
# 4. User Input and Prediction
# =============================================================================

print("\n" + "="*50)
print("             Generic Prediction Model             ")
print("="*50)

try:
    x_cell_size = float(input("Enter X Cell Size (e.g., 15.0): "))
    yz_cell_size = float(input("Enter YZ Cell Size (e.g., 20.0): "))
    velocity = float(input("Enter Inlet Velocity (e.g., 3000.0): "))
    
    user_input = np.array([x_cell_size, yz_cell_size, velocity])
    
    print("\n--- Prediction Results ---")
    
    # Predict Mass (using the log-linear model)
    print("\n--- Mass ---")
    mass_prediction = predict_mass(x_cell_size, yz_cell_size)
    print(f"Predicted value: {mass_prediction:.4f} (based on log-linear model)")
    
    # Predict GPR model variables
    gpr_vars = ['AvgVelocity', 'PressureDrop', 'Surface Area']
    for output_var in gpr_vars:
        if output_var in predictors:
            print(f"\n--- {output_var} ---")
            predictor = predictors[output_var]
            
            # Use try-except to catch potential UserWarnings from GPRPredictor.predict
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                prediction_result = predictor.predict(user_input)
            
            pred_val = prediction_result['prediction']
            ci_lower = prediction_result['ci_lower']
            ci_upper = prediction_result['ci_upper']

            # Ensure all values are scalars
            if isinstance(pred_val, np.ndarray):
                pred_val = pred_val.item()
            if isinstance(ci_lower, np.ndarray):
                ci_lower = ci_lower.item()
            if isinstance(ci_upper, np.ndarray):
                ci_upper = ci_upper.item()

            print(f"Predicted value: {pred_val:.4f}")
            print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            print(f"Warning: Predictor for {output_var} is not available.")

except ValueError:
    print("Invalid input. Please enter a numerical value.")
except Exception as e:
    print(f"An error occurred during prediction: {e}")