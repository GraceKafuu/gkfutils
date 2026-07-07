import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def calculate_curvature(x, y, smooth_sigma=1.0, use_smooth=True):
    """
    计算离散点集的曲率
    
    参数:
        x: 数组, x坐标点
        y: 数组, y坐标点
        smooth_sigma: 高斯平滑标准差，越大越平滑
        use_smooth: 是否启用平滑（推荐True，减少噪声）
    
    返回:
        curvature: 各点的曲率数组
        curvature_radius: 各点的曲率半径数组
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 【可选】高斯平滑，减少噪声对导数的影响
    if use_smooth:
        x = gaussian_filter1d(x, sigma=smooth_sigma)
        y = gaussian_filter1d(y, sigma=smooth_sigma)
    
    # 1. 计算一阶导数（中心差分法）
    dx = np.gradient(x)  # x' = dx/dt
    dy = np.gradient(y)  # y' = dy/dt
    
    # 2. 计算二阶导数
    d2x = np.gradient(dx)  # x''
    d2y = np.gradient(dy)  # y''
    
    # 3. 分子：|x'y'' - x''y'|
    numerator = np.abs(dx * d2y - d2x * dy)
    
    # 4. 分母：(x'² + y'²)^(3/2)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # 5. 避免除以零（直线/静止点）
    denominator[denominator < 1e-10] = 1e-10
    
    # 6. 计算曲率和曲率半径
    curvature = numerator / denominator
    curvature_radius = 1.0 / (curvature + 1e-10)  # 加小值避免无穷大
    
    return curvature, curvature_radius


# ====================== 示例测试 ======================
if __name__ == "__main__":
    # 生成测试曲线：正弦曲线 + 噪声
    t = np.linspace(0, 4 * np.pi, 200)  # 参数t
    x = t
    y = np.sin(t) + 0.02 * np.random.randn(len(t))  # 加少量噪声
    
    # 计算曲率
    curvature, curvature_radius = calculate_curvature(
        x, y, 
        smooth_sigma=2.0, 
        use_smooth=True
    )
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 原始曲线
    ax1.plot(x, y, 'b-', linewidth=2, label='原始曲线')
    ax1.scatter(x, y, c=curvature, cmap='jet', s=30, label='曲率着色')
    ax1.set_title('原始曲线（颜色表示曲率大小）', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 曲率变化
    ax2.plot(t, curvature, 'r-', linewidth=2, label='曲率 K')
    ax2.set_title('各点曲率变化', fontsize=14)
    ax2.set_xlabel('参数 t')
    ax2.set_ylabel('曲率 K')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.colorbar(ax1.collections[0], ax=ax1, label='曲率值')
    plt.tight_layout()
    plt.show()
    
    # 输出关键信息
    print(f"✅ 计算完成！")
    print(f"📊 最大曲率: {np.max(curvature):.4f} (在 t={t[np.argmax(curvature)]:.2f})")
    print(f"📊 最小曲率: {np.min(curvature):.4f}")
    print(f"📊 平均曲率: {np.mean(curvature):.4f}")
    print(f"📊 最大曲率半径: {np.max(curvature_radius):.2f}")