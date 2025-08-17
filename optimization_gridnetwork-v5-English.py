import pandas as pd
import numpy as np
import json
import warnings
from scipy.optimize import minimize, Bounds
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, ConstantKernel, WhiteKernel
from itertools import product

# Ignore all warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. Data Loading and Model Parameter Preparation
# =============================================================================
file_path_train = "LHS625.csv"
file_path_results = "universal_kriging_all_results_LHS625_Scalerv3.json"

try:
    df_train = pd.read_csv(file_path_train, skipinitialspace=True)
    with open(file_path_results, 'r', encoding='utf-8') as f:
        all_results_data = json.load(f)['results']
    print("Data and model parameters loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find the data or results file. Please check the paths.")
    exit()

input_vars = ['X Cell Size', 'YZ Cell Size', 'Velocity Inlet']
df_train_inputs = df_train[input_vars]

# --- Define the search boundaries for the variables ---
bounds = Bounds([10, 10, 2500], [25, 25, 3500])

# =============================================================================
# 2. Prediction Model Implementation
# =============================================================================
class GPRPredictor:
    def __init__(self, model_params, X_train, y_train):
        self.scaler_X = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.scaler_y = StandardScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        self.poly = PolynomialFeatures(degree=model_params['poly_degree'])
        self.X_poly_train = self.poly.fit_transform(self.X_train_scaled)
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_poly_train, self.y_train_scaled)
        
        kernel_choice = model_params['kernel_type']
        params = model_params['gpr_optimized_params']
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
                length_scale_val, periodicity_val = length_scales[0], length_scales[1]
            else:
                length_scale_val, periodicity_val = length_scales, length_scales
            kernel = ConstantKernel(constant_value) * ExpSineSquared(length_scale=length_scale_val, periodicity=periodicity_val)
        
        kernel += WhiteKernel(noise_level=noise_level)
        self.gpr_model = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=False)
        residuals_train = self.y_train_scaled - self.linear_model.predict(self.X_poly_train)
        self.gpr_model.fit(self.X_train_scaled, residuals_train)

    def predict(self, X_new_df):
        X_new_scaled = self.scaler_X.transform(X_new_df)
        X_poly_new = self.poly.transform(X_new_scaled)
        y_trend_new_scaled = self.linear_model.predict(X_poly_new)
        y_residuals_pred_scaled = self.gpr_model.predict(X_new_scaled)
        y_pred_scaled = y_trend_new_scaled + y_residuals_pred_scaled
        y_pred_orig = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred_orig

# --- Physics Model Functions (Mass and Surface Area) ---
def predict_log_linear(X_val, YZ_val, coeffs):
    if X_val <= 0 or YZ_val <= 0: return 1e10
    logX, logYZ = np.log(X_val), np.log(YZ_val)
    log_pred = (coeffs['const'] + coeffs['logX'] * logX + coeffs['logYZ'] * logYZ +
                coeffs['logX_sq'] * logX**2 + coeffs['logYZ_sq'] * logYZ**2 +
                coeffs['logX_logYZ'] * logX * logYZ)
    return np.exp(log_pred)

# Updated Mass and Surface Area coefficients
mass_coeffs = {
    'const': 6.2696, 'logX': -0.2222, 'logYZ': -0.4861,
    'logX_sq': 0.0085, 'logYZ_sq': 0.0395, 'logX_logYZ': 0.0292
}
sa_coeffs = {'const': 12.0430, 'logX': -0.2628, 'logYZ': -0.5268, 'logX_sq': 0.1427, 'logYZ_sq': 0.1597, 'logX_logYZ': -0.2592}

# --- Instantiate the specified GPR predictors ---
fixed_models = {
    'AvgVelocity': {'poly_degree': 3, 'kernel_type': 'Matern1.5'},
    'PressureDrop': {'poly_degree': 3, 'kernel_type': 'RBF'},
    'Surface Area': {'poly_degree': 3, 'kernel_type': 'Matern1.5'},
}

try:
    gpr_models = {}
    for output_var, fixed_params in fixed_models.items():
        poly_degree_val = fixed_params['poly_degree']
        kernel_type = fixed_params['kernel_type']
        poly_degree_key = f"poly_{poly_degree_val}"

        model_params = all_results_data.get(output_var, {}).get(poly_degree_key, {}).get(kernel_type)
        
        if not model_params:
            print(f"Warning: Could not find the specified model for {output_var} (polynomial={poly_degree_val}, kernel={kernel_type}).")
            raise ValueError(f"Missing model parameters for {output_var}")

        gpr_models[output_var] = GPRPredictor(model_params, df_train_inputs, df_train[output_var])

    gpr_sa = gpr_models['Surface Area']
    gpr_pressure = gpr_models['PressureDrop']
    gpr_avg_vel = gpr_models['AvgVelocity']
except Exception as e:
    print(f"Failed to instantiate GPR predictors: {e}")
    exit()

# =============================================================================
# 3. Multi-Start Local Optimization
# =============================================================================

print("\n" + "="*20, "Starting Multi-Start Local Optimization", "="*20)

# Define multiple starting points
start_points_x = [10, 15, 20, 25]
start_points_yz = [10, 15, 20, 25]
start_points_v = [2500, 3000, 3500]

# Generate all combinations of starting points
start_points = list(product(start_points_x, start_points_yz, start_points_v))

print(f"Generated {len(start_points)} optimization starting points.")

# Objective function and constraints
def objective_fn(p):
    sa_pred = gpr_sa.predict(np.array([[p[0], p[1], p[2]]]))
    return -sa_pred[0]

slsqp_constraints = [
    {'type': 'ineq', 'fun': lambda p: 125 - predict_log_linear(p[0], p[1], mass_coeffs)},
    {'type': 'ineq', 'fun': lambda p: 8000 - gpr_pressure.predict(np.array([[p[0], p[1], p[2]]]))[0]},
    {'type': 'ineq', 'fun': lambda p: gpr_avg_vel.predict(np.array([[p[0], p[1], p[2]]]))[0] - 520}
]

# Run local optimization from each starting point
local_optima = []
for i, start_point in enumerate(start_points):
    result = minimize(objective_fn, np.array(start_point), method='SLSQP', bounds=bounds, constraints=slsqp_constraints, options={'disp': False})
    if result.success:
        local_optima.append(result)

# Summarize all local optima
if not local_optima:
    print("\n--- Warning: No local optima satisfying the constraints were found. ---")
else:
    best_result = min(local_optima, key=lambda res: res.fun)
    
    opt_vars = best_result.x
    x_opt, yz_opt, v_opt = opt_vars[0], opt_vars[1], opt_vars[2]
    
    final_sa = -best_result.fun
    final_mass = predict_log_linear(x_opt, yz_opt, mass_coeffs)
    final_pressure = gpr_pressure.predict(np.array([[x_opt, yz_opt, v_opt]]))[0]
    final_avg_vel = gpr_avg_vel.predict(np.array([[x_opt, yz_opt, v_opt]]))[0]

    print("\n--- Multi-Start Local Optimization Complete ---")
    print(f"Found {len(local_optima)} local optima satisfying the constraints.")
    print(f"Global Optimal X Cell Size: {x_opt:.4f} mm")
    print(f"Global Optimal YZ Cell Size: {yz_opt:.4f} mm")
    print(f"Global Optimal Inlet Velocity: {v_opt:.4f} mm/s")
    print(f"Maximum Surface Area Achieved: {final_sa:.4f} mm²")
    
    print("\n--- Constraint Values at the Optimal Point ---")
    print(f"Mass: {final_mass:.4f} g (Constraint: < 125 g)")
    print(f"Pressure: {final_pressure:.4f} Pa (Constraint: < 8000 Pa)")
    print(f"Average Velocity: {final_avg_vel:.4f} mm/s (Constraint: > 520 mm/s)")