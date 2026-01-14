import numpy as np
import matplotlib.pyplot as plt

def curvature_explicit(x, a, b, c):
    """
    计算抛物线 y = ax² + bx + c 在各点的曲率
    
    参数:
        x: x坐标值（标量或数组）
        a, b, c: 抛物线系数 y = ax² + bx + c
    
    返回:
        各点对应的曲率值
    """
    # 一阶导数: y' = 2ax + b
    y_prime = 2 * a * x + b
    
    # 二阶导数: y'' = 2a
    y_double_prime = 2 * a
    
    # 曲率公式
    numerator = np.abs(y_double_prime)
    denominator = (1 + y_prime**2)**(3/2)
    
    return numerator / denominator

# # 示例：抛物线 y = 0.1x²
# x_values = np.linspace(-10, 10, 100)
# a, b, c = 0.1, 0, 0
# curvatures = curvature_explicit(x_values, a, b, c)

# # 可视化
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.plot(x_values, a*x_values**2 + b*x_values + c, 'b-', linewidth=2)
# plt.title('抛物线: y = 0.1x²')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(x_values, curvatures, 'r-', linewidth=2)
# plt.title('曲率分布')
# plt.xlabel('x')
# plt.ylabel('曲率 κ')
# plt.grid(True)

# plt.tight_layout()
# plt.show()


def curvature_parametric(x, y, t=None):
    """
    使用参数方程形式计算曲率（最通用的方法）
    
    参数:
        x, y: 曲线点的坐标
        t: 参数（如果为None，则使用弧长近似）
    
    返回:
        各点曲率值
    """
    if t is None:
        # 使用累积弦长作为参数近似
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.sqrt(dx**2 + dy**2)
        t = np.concatenate(([0], np.cumsum(dt)))
    
    # 数值微分计算导数
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)
    
    # 参数形式的曲率公式
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)
    
    # 避免除零
    curvature = np.zeros_like(numerator)
    mask = denominator > 1e-10  # 避免除零
    curvature[mask] = numerator[mask] / denominator[mask]
    
    return curvature

# # 示例：抛物线参数方程
# t_param = np.linspace(-2, 2, 100)
# x_param = t_param
# y_param = 0.1 * t_param**2

# curvature_param = curvature_parametric(x_param, y_param, t_param)

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.plot(x_param, y_param, 'g-', linewidth=2)
# plt.title('参数形式抛物线')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)

# plt.subplot(1, 3, 2)
# plt.plot(t_param, curvature_param, 'm-', linewidth=2)
# plt.title('参数形式计算的曲率')
# plt.xlabel('参数 t')
# plt.ylabel('曲率 κ')
# plt.grid(True)

# plt.subplot(1, 3, 3)
# # 用颜色表示曲率大小
# scatter = plt.scatter(x_param, y_param, c=curvature_param, cmap='viridis', s=50)
# plt.colorbar(scatter, label='曲率')
# plt.title('曲率可视化')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)

# plt.tight_layout()
# plt.show()


def analyze_parabola_curvature(a, b, c, x_range=(-10, 10), num_points=200):
    """
    完整分析抛物线的曲率特性
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = a * x**2 + b * x + c
    
    # 计算曲率
    curvature = curvature_explicit(x, a, b, c)
    
    # 找到最小曲率点（抛物线顶点）
    min_curvature_idx = np.argmin(curvature)
    min_curvature_x = x[min_curvature_idx]
    min_curvature_value = curvature[min_curvature_idx]
    
    # 找到最大曲率点
    max_curvature_idx = np.argmax(curvature)
    max_curvature_x = x[max_curvature_idx]
    max_curvature_value = curvature[max_curvature_idx]
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制曲线和曲率点
    ax1.plot(x, y, 'b-', linewidth=2, label=f'y = {a}x² + {b}x + {c}')
    ax1.plot(min_curvature_x, y[min_curvature_idx], 'ro', 
             markersize=8, label=f'最小曲率点 ({min_curvature_x:.2f}, {y[min_curvature_idx]:.2f})')
    ax1.plot(max_curvature_x, y[max_curvature_idx], 'go', 
             markersize=8, label=f'最大曲率点 ({max_curvature_x:.2f}, {y[max_curvature_idx]:.2f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('抛物线')
    
    # 绘制曲率分布
    ax2.plot(x, curvature, 'r-', linewidth=2, label='曲率')
    ax2.plot(min_curvature_x, min_curvature_value, 'ro', markersize=8)
    ax2.plot(max_curvature_x, max_curvature_value, 'go', markersize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('曲率 κ')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('曲率分布')
    
    plt.tight_layout()
    plt.show()
    
    print(f"抛物线方程: y = {a}x² + {b}x + {c}")
    print(f"最小曲率: {min_curvature_value:.6f} (在 x = {min_curvature_x:.2f})")
    print(f"最大曲率: {max_curvature_value:.6f} (在 x = {max_curvature_x:.2f})")
    print(f"曲率范围: [{curvature.min():.6f}, {curvature.max():.6f}]")
    
    return x, y, curvature

# 测试不同抛物线
print("=== 抛物线 y = 0.1x² ===")
x1, y1, k1 = analyze_parabola_curvature(0.1, 0, 0)

print("\n=== 抛物线 y = 0.5x² + 2x + 1 ===")
x2, y2, k2 = analyze_parabola_curvature(0.5, 2, 1)