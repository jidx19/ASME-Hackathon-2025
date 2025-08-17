# 标准化后预测，gp模型

import pandas as pd
import numpy as np
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, RationalQuadratic, DotProduct, ConstantKernel, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. 数据加载 ---
file_path_test = "nTop ASME Hackathon Data.csv"
file_path_train = "LHS625.csv"
try:
    df_train = pd.read_csv(file_path_train, skipinitialspace=True)
    df_test = pd.read_csv(file_path_test, skipinitialspace=True)
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：文件路径不正确，使用虚拟数据进行演示。")
    np.random.seed(42)
    df_train = pd.DataFrame(np.random.rand(500, 8) * 100, columns=['AvgVelocity', 'Mass', 'PressureDrop', 'Surface Area', 'Velocity Inlet', 'X Cell Size', 'YZ Cell Size', 'Dummy'])
    df_test = pd.DataFrame(np.random.rand(20, 8) * 100, columns=['AvgVelocity', 'Mass', 'PressureDrop', 'Surface Area', 'Velocity Inlet', 'X Cell Size', 'YZ Cell Size', 'Dummy'])

# --- 2. GPR核函数构建函数 (已更新) ---
def build_gpr_model(X_train_shape, kernel_choice='RBF', optimize_params=True):
    """
    构建高斯过程回归模型核函数。
    """
    n_features = X_train_shape[1]
    
    if kernel_choice == 'RBF':
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e5))
    elif kernel_choice == 'Matern1.5':
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e5), nu=1.5)
    elif kernel_choice == 'Matern2.5':
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e5), nu=2.5)
    elif kernel_choice == 'Periodic':
        # Periodic 核对输入维度不敏感，因为它只依赖于长度和周期性
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
    elif kernel_choice == 'RQ':
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=[1.0] * n_features, alpha=1.0)
    elif kernel_choice == 'Composite':
        # 组合核需要分开处理
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * n_features) \
               + ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0)
    else:
        raise ValueError(f"Unsupported kernel choice: {kernel_choice}")

    kernel += WhiteKernel(noise_level=0.001, noise_level_bounds=(1e-5, 1e1))
    
    gpr_model = GaussianProcessRegressor(
        kernel=kernel,
        optimizer='fmin_l_bfgs_b' if optimize_params else None,
        n_restarts_optimizer=10 if optimize_params else 0,
        random_state=42
    )
    return gpr_model

# --- 3. 通用克里金模型训练函数 (已修复残差计算) ---
def train_and_evaluate_model(X_train, y_train, X_test, y_test, poly_degree, kernel_choice):
    """
    通用克里金模型训练、预测和评估（已修复残差计算）
    """
    # 标准化输入特征
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 标准化输出变量
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # 趋势项建模（在标准化后的数据上）
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly_train = poly.fit_transform(X_train_scaled)
    X_poly_test = poly.transform(X_test_scaled)
    linear_model = LinearRegression()
    linear_model.fit(X_poly_train, y_train_scaled)
    y_trend_train = linear_model.predict(X_poly_train)
    y_trend_test = linear_model.predict(X_poly_test)
    
    # 残差建模
    residuals_train = y_train_scaled - y_trend_train
    gpr_model = build_gpr_model(X_train_scaled.shape, kernel_choice=kernel_choice)
    gpr_model.fit(X_train_scaled, residuals_train)
    
    # 预测（在标准化空间）
    y_residuals_pred, y_residuals_std = gpr_model.predict(X_test_scaled, return_std=True)
    y_pred_scaled = y_trend_test + y_residuals_pred
    
    # 反标准化预测结果
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_trend_test_orig = scaler_y.inverse_transform(y_trend_test.reshape(-1, 1)).ravel()
    
    # 关键修复：正确计算残差预测值（原始空间）
    # 残差预测值 = 总预测值 - 趋势预测值
    y_residuals_pred_orig = y_pred - y_trend_test_orig
    
    # 计算标准差在原始空间的值
    y_residuals_std_orig = y_residuals_std * scaler_y.scale_
    
    # 验证关系：趋势 + 残差 = 预测值
    # 添加验证逻辑确保关系成立
    for i in range(len(y_pred)):
        # 计算允许的浮点数误差范围
        tolerance = 1e-5
        computed_sum = y_trend_test_orig[i] + y_residuals_pred_orig[i]
        
        # 验证关系是否成立
        if abs(computed_sum - y_pred[i]) > tolerance:
            print(f"警告: 在索引{i}处，趋势({y_trend_test_orig[i]}) + 残差({y_residuals_pred_orig[i]}) 不等于预测({y_pred[i]})")
    
    # 评估（在原始空间）
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # 提取优化后的核参数
    try:
        if kernel_choice in ['RBF', 'Matern1.5', 'Matern2.5', 'RQ']:
            length_scales = gpr_model.kernel_.k1.k2.get_params()['length_scale']
        elif kernel_choice == 'Periodic':
            length_scales = gpr_model.kernel_.k1.k2.get_params()['length_scale']
        elif kernel_choice == 'Composite':
            k1_params = gpr_model.kernel_.k1.k1.get_params()
            k2_params = gpr_model.kernel_.k1.k2.get_params()
            length_scales = {'rbf_length': k1_params['k2__length_scale'], 'dot_sigma_0': k2_params['k2__sigma_0']}
        else:
            length_scales = None
    except Exception:
        length_scales = "Error"
        
    gpr_params = {
        'length_scales': length_scales,
        'constant_value': gpr_model.kernel_.k1.k1.get_params()['constant_value'] if hasattr(gpr_model.kernel_, 'k1') else None,
        'noise_level': gpr_model.kernel_.k2.get_params()['noise_level'] if hasattr(gpr_model.kernel_, 'k2') else gpr_model.kernel_.get_params()['noise_level']
    }

    # 保存每个测试点的预测值和标准差（原始空间）
    test_predictions = []
    for i in range(len(X_test)):
        test_predictions.append({
            'x': X_test.iloc[i].values.tolist() if isinstance(X_test, pd.DataFrame) else X_test[i].tolist(),
            'true_value': y_test.iloc[i] if isinstance(y_test, pd.Series) else y_test[i],
            'prediction': y_pred[i],
            'std_dev': y_residuals_std_orig[i],
            'trend_prediction': y_trend_test_orig[i],
            'residual_prediction': y_residuals_pred_orig[i]
        })
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'poly_degree': poly_degree,
        'kernel_type': kernel_choice,
        'trend_coeffs': linear_model.coef_.tolist(),
        'trend_intercept': linear_model.intercept_,
        'gpr_optimized_params': gpr_params,
        'test_predictions': test_predictions,
        'scaler_y_mean': scaler_y.mean_[0],
        'scaler_y_scale': scaler_y.scale_[0]
    }

# 修改后的结果保存函数
def save_results_to_json(results, output_file='universal_kriging_all_results_LHS625_Scalerv3.json'):
    """
    将所有模型训练结果保存到JSON文件（包含测试点预测和标准差）
    """
    print(f"\n=== 保存完整结果到JSON文件 ===")
    
    save_data = {
        'model_info': {
            'model_type': 'Universal Kriging with StdDev (Standardized)',
            'total_experiments': sum(len(kernels) for poly_degree_res in results.values() for kernels in poly_degree_res.values()),
            'timestamp': pd.Timestamp.now().isoformat()
        },
        'results': results
    }
    
    # 确保所有数据可序列化
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print(f"完整结果已成功保存到: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    return True

# --- 5. 核心训练循环和结果汇总 ---
input_vars = ['X Cell Size', 'YZ Cell Size', 'Velocity Inlet']
output_vars = ['AvgVelocity', 'Mass', 'PressureDrop', 'Surface Area']
poly_degrees = [0, 1, 2, 3]
kernel_choices = ['RBF', 'Matern1.5', 'Matern2.5','Periodic']

all_results = {}

print("开始进行通用克里金全量建模（已添加标准化处理）...\n")

# 循环遍历所有组合
for output_var in output_vars:
    all_results[output_var] = {}
    
    y_train = df_train[output_var]
    y_test = df_test[output_var]
    
    for poly_degree in poly_degrees:
        all_results[output_var][f'poly_{poly_degree}'] = {}
        
        for kernel_choice in kernel_choices:
            print(f"--> 正在处理: {output_var}, 多项式={poly_degree}, 核函数={kernel_choice}")
            try:
                result = train_and_evaluate_model(
                    df_train[input_vars], y_train, df_test[input_vars], y_test,
                    poly_degree, kernel_choice
                )
                all_results[output_var][f'poly_{poly_degree}'][kernel_choice] = result
            except Exception as e:
                print(f"    - 模型训练失败: {str(e)}")
                all_results[output_var][f'poly_{poly_degree}'][kernel_choice] = {
                    'RMSE': 'Error',
                    'MAE': 'Error',
                    'trend_coeffs': 'Error',
                    'gpr_optimized_params': 'Error'
                }
            
            # 打印指标（如果可用）
            res_dict = all_results[output_var][f'poly_{poly_degree}'][kernel_choice]
            if isinstance(res_dict, dict) and 'RMSE' in res_dict and isinstance(res_dict['RMSE'], float):
                print(f"    - RMSE: {res_dict['RMSE']:.4f}")
                print(f"    - MAE: {res_dict['MAE']:.4f}\n")
            else:
                print("    - RMSE: 计算失败\n    - MAE: 计算失败\n")

# --- 6. 结果汇总表格 ---
print("\n" + "="*80)
print("                    所有模型预测效果汇总表格 (已标准化)                   ")
print("="*80)

for output_var in output_vars:
    print(f"\n### 目标变量: {output_var}")
    
    header = f"{'趋势项次数':<10}"
    for kernel in kernel_choices:
        header += f"| {kernel:>10} RMSE | {kernel:>10} MAE "
    print(header)
    print("-" * 80)
    
    for poly_degree in poly_degrees:
        row = f"{poly_degree:<10}"
        for kernel in kernel_choices:
            result = all_results[output_var][f'poly_{poly_degree}'][kernel]
            if isinstance(result, dict) and isinstance(result.get('RMSE'), float):
                row += f"| {result['RMSE']:>10.4f} | {result['MAE']:>10.4f} "
            else:
                row += f"| {'Error':>10} | {'Error':>10} "
        print(row)
    print("-" * 80)

# --- 7. 保存所有结果到JSON文件 ---
save_results_to_json(all_results)
print("所有建模完成！")