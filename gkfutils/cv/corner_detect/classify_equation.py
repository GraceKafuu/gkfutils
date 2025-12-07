import numpy as np
import matplotlib.pyplot as plt

# ====================== 基础函数定义 ======================
def quartic_func(x, coeffs):
    """计算一元四次函数值：f(x) = a*x⁴ + b*x³ + c*x² + d*x + e"""
    a, b, c, d, e = coeffs
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def first_deriv(x, coeffs):
    """一阶导数：f'(x) = 4a*x³ + 3b*x² + 2c*x + d"""
    a, b, c, d, _ = coeffs
    return 4*a * x**3 + 3*b * x**2 + 2*c * x + d

def second_deriv(x, coeffs):
    """二阶导数：f''(x) = 12a*x² + 6b*x + 2c"""
    a, b, c, _, _ = coeffs
    return 12*a * x**2 + 6*b * x + 2*c

# ====================== 极值点计算（一阶+二阶导数） ======================
def get_extrema_points(coeffs, x_min, x_max, eps=1e-6):
    """
    计算区间内的极值点（x坐标、y坐标、极值类型）
    参数：
        coeffs: 四次方程系数 [a,b,c,d,e]
        x_min/x_max: 区间边界
        eps: 数值精度容错
    返回：
        extrema_list: 极值点列表，每个元素为 (x, y, type)
                      type: 'min'（极小值）/'max'（极大值）
        extrema_x: 仅x坐标列表
        extrema_y: 仅y坐标列表
    """
    # 1. 求解一阶导数（三次方程）的根 → 驻点
    a, b, c, d, _ = coeffs
    deriv1_coeffs = [4*a, 3*b, 2*c, d]
    roots = np.roots(deriv1_coeffs)

    # 2. 过滤实数根 + 去重 + 区间筛选
    real_roots = []
    for root in roots:
        if np.isreal(root) or abs(np.imag(root)) < eps:
            real_roots.append(np.round(np.real(root), 6))
    real_roots = list(np.unique(real_roots))  # 去重
    stationary_pts = [x for x in real_roots if (x_min - eps) <= x <= (x_max + eps)]

    # 3. 用二阶导数判定极值点（f''(x)≠0）
    extrema_list = []
    extrema_x = []
    extrema_y = []
    for x0 in stationary_pts:
        f2 = second_deriv(x0, coeffs)
        if abs(f2) < eps:  # 二阶导数为0 → 拐点（非极值点）
            continue
        
        # 计算极值点y值
        y0 = np.round(quartic_func(x0, coeffs), 6)
        # 判断极值类型
        extrema_type = 'min' if f2 > 0 else 'max'
        
        extrema_list.append((x0, y0, extrema_type))
        extrema_x.append(x0)
        extrema_y.append(y0)

    return extrema_list, extrema_x, extrema_y

# ====================== 分类逻辑（按极值点数量+Y差值） ======================
def classify_equation(extrema_y, threshold=1.0):
    """
    按规则分类：
    - 1个极值点 → 0
    - ≥2个极值点：
      - Y差值绝对值>阈值 → 1
      - Y差值绝对值≤阈值 → 2
    """
    count = len(extrema_y)
    if count == 1:
        return 0, f"极值点数量={count} → 类别0"
    elif count >= 2:
        y_max = max(extrema_y)
        y_min = min(extrema_y)
        y_diff = abs(y_max - y_min)
        if y_diff > threshold:
            return 2, f"极值点数量={count}，Y差值={y_diff:.2f} > 阈值{threshold} → 类别1"
        else:
            return 1, f"极值点数量={count}，Y差值={y_diff:.2f} ≤ 阈值{threshold} → 类别2"
    else:
        return -1, "无极值点 → 无类别"

# ====================== 可视化（标注极值点+分类结果） ======================
def plot_curve(coeffs, x_min, x_max, extrema_list, classify_info, threshold):
    """绘制函数曲线，标注极值点、Y差值、分类结果"""
    # 生成绘图数据
    x_plot = np.linspace(x_min - 1, x_max + 1, 2000)
    y_plot = quartic_func(x_plot, coeffs)

    # 提取极值点坐标和类型
    extrema_x = [p[0] for p in extrema_list]
    extrema_y = [p[1] for p in extrema_list]
    extrema_types = [p[2] for p in extrema_list]

    # 绘图设置
    plt.figure(figsize=(14, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制函数曲线
    plt.plot(x_plot, y_plot, color='#2E86AB', linewidth=2, label='四次函数曲线')

    # 标注极值点（区分极大/极小）
    if extrema_list:
        # 极小值点（绿色实心）
        min_x = [x for x, _, t in extrema_list if t == 'min']
        min_y = [y for _, y, t in extrema_list if t == 'min']
        if min_x:
            plt.scatter(min_x, min_y, color='#4CAF50', s=150, 
                        label='极小值点', zorder=5, marker='o')
            for x, y in zip(min_x, min_y):
                plt.annotate(f'极小值\n({x:.2f}, {y:.2f})', (x, y), 
                             xytext=(5, -15), textcoords='offset points', fontsize=9)
        
        # 极大值点（红色实心）
        max_x = [x for x, _, t in extrema_list if t == 'max']
        max_y = [y for _, y, t in extrema_list if t == 'max']
        if max_x:
            plt.scatter(max_x, max_y, color='#F44336', s=150, 
                        label='极大值点', zorder=5, marker='^')
            for x, y in zip(max_x, max_y):
                plt.annotate(f'极大值\n({x:.2f}, {y:.2f})', (x, y), 
                             xytext=(5, 10), textcoords='offset points', fontsize=9)
        
        # 标注Y差值
        if len(extrema_y) >= 2:
            y_max = max(extrema_y)
            y_min = min(extrema_y)
            y_diff = abs(y_max - y_min)
            plt.text(0.02, 0.95, f'Y差值绝对值：{y_diff:.2f}\n阈值：{threshold}', 
                     transform=plt.gca().transAxes, fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))

    # 标注区间和分类结果
    plt.axvspan(x_min, x_max, alpha=0.15, color='#70D6FF', label=f'目标区间 [{x_min}, {x_max}]')
    plt.text(0.02, 0.85, classify_info, transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))

    # 坐标轴与网格
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    # 图表标签
    plt.xlabel('x 轴', fontsize=12)
    plt.ylabel('y = f(x)', fontsize=12)
    plt.title('一元四次函数极值点分析（一阶+二阶导数）', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()

# ====================== 主程序 + 测试案例 ======================
if __name__ == "__main__":
    # 可调参数：Y差值阈值
    Y_THRESHOLD = 100.0

    # ---------------------- 测试案例1：3个极值点，Y差值>阈值 → 类别1 ----------------------
    print("="*70)
    print("测试案例1：f(x) = x⁴ - 2x² + 1（Y差值=1.0 > 阈值1.0 → 类别1）")
    coeffs1 = [1, 0, -2, 0, 1]
    x_min1, x_max1 = -2, 2
    extrema1, ext_x1, ext_y1 = get_extrema_points(coeffs1, x_min1, x_max1)
    class1, class_info1 = classify_equation(ext_y1, Y_THRESHOLD)
    print(f"极值点详情：{extrema1}")
    print(f"分类结果：{class_info1}")
    plot_curve(coeffs1, x_min1, x_max1, extrema1, class_info1, Y_THRESHOLD)

    # ---------------------- 测试案例2：2个极值点，Y差值≤阈值 → 类别2 ----------------------
    print("\n" + "="*70)
    print("测试案例2：f(x) = x⁴ - 0.5x² + 0.2（Y差值≈0.2 < 阈值1.0 → 类别2）")
    coeffs2 = [1, 0, -0.5, 0, 0.2]
    x_min2, x_max2 = -2, 2
    extrema2, ext_x2, ext_y2 = get_extrema_points(coeffs2, x_min2, x_max2)
    class2, class_info2 = classify_equation(ext_y2, Y_THRESHOLD)
    print(f"极值点详情：{extrema2}")
    print(f"分类结果：{class_info2}")
    plot_curve(coeffs2, x_min2, x_max2, extrema2, class_info2, Y_THRESHOLD)

    # ---------------------- 测试案例3：1个极值点 → 类别0 ----------------------
    print("\n" + "="*70)
    print("测试案例3：f(x) = x⁴ + 2x³ + 3x² + 4x + 5（1个极值点 → 类别0）")
    coeffs3 = [1, 2, 3, 4, 5]
    x_min3, x_max3 = -3, 1
    extrema3, ext_x3, ext_y3 = get_extrema_points(coeffs3, x_min3, x_max3)
    class3, class_info3 = classify_equation(ext_y3, Y_THRESHOLD)
    print(f"极值点详情：{extrema3}")
    print(f"分类结果：{class_info3}")
    plot_curve(coeffs3, x_min3, x_max3, extrema3, class_info3, Y_THRESHOLD)


    # ---------------------- 测试案例3：1个极值点 → 类别0 ----------------------
    print("\n" + "="*70)
    # print("测试案例3：f(x) = x⁴ + 2x³ + 3x² + 4x + 5（1个极值点 → 类别0）")
    coeffs3 = [-3.81730120e-10,  6.32472159e-07, -3.44592126e-04,  6.36720956e-02, 9.74446751e+01]
    x_min3, x_max3 = 0, 2000
    extrema3, ext_x3, ext_y3 = get_extrema_points(coeffs3, x_min3, x_max3)
    class3, class_info3 = classify_equation(ext_y3, Y_THRESHOLD)
    print(f"极值点详情：{extrema3}")
    print(f"分类结果：{class_info3}")
    plot_curve(coeffs3, x_min3, x_max3, extrema3, class_info3, Y_THRESHOLD)