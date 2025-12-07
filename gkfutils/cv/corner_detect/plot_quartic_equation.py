import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def plot_quartic_equation(a, b, c, d, e, interval_start, interval_end, 
                         num_points=1000, show_extremum=True, show_grid=True):
    """
    绘制一元四次方程在指定区间的图像
    
    参数:
    a, b, c, d, e: 四次方程 ax^4 + bx^3 + cx^2 + dx + e 的系数
    interval_start, interval_end: 区间起点和终点
    num_points: 绘图使用的点数
    show_extremum: 是否显示极值点
    show_grid: 是否显示网格
    
    返回:
    fig: 图形对象
    ax: 坐标轴对象
    """
    
    # 1. 定义函数
    def f(x):
        return a*x**4 + b*x**3 + c*x**2 + d*x + e
    
    # 2. 定义一阶导数
    def f_prime(x):
        return 4*a*x**3 + 3*b*x**2 + 2*c*x + d
    
    # 3. 定义二阶导数
    def f_double_prime(x):
        return 12*a*x**2 + 6*b*x + 2*c
    
    # 4. 创建x轴数据
    x = np.linspace(interval_start, interval_end, num_points)
    y = f(x)
    
    # 5. 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 6. 绘制函数曲线
    ax.plot(x, y, 'b-', linewidth=2, label=f'$f(x) = {format_coeff(a)}x^4 {format_coeff(b, True)}x^3 {format_coeff(c, True)}x^2 {format_coeff(d, True)}x {format_coeff(e, True)}$')
    
    # 7. 寻找并标记极值点
    extremum_points = []
    if show_extremum:
        # 在区间内采样多个点作为寻找根的初始猜测
        num_initial_guesses = 20
        guesses = np.linspace(interval_start, interval_end, num_initial_guesses)
        
        found_roots = set()
        
        for guess in guesses:
            try:
                # 使用数值方法寻找f'(x)=0的根
                sol = root_scalar(f_prime, x0=guess, bracket=[interval_start, interval_end])
                
                if sol.converged:
                    x_root = sol.root
                    
                    # 去重处理
                    is_duplicate = False
                    for existing_root in found_roots:
                        if abs(x_root - existing_root) < 1e-6:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and interval_start <= x_root <= interval_end:
                        found_roots.add(x_root)
                        
                        # 检查是否为极值点
                        second_derivative = f_double_prime(x_root)
                        y_root = f(x_root)
                        
                        if second_derivative != 0:
                            extremum_points.append((x_root, y_root, second_derivative))
                        else:
                            # 使用一阶导数符号变化测试
                            epsilon = 1e-6
                            left_val = f_prime(x_root - epsilon)
                            right_val = f_prime(x_root + epsilon)
                            
                            if left_val * right_val < 0:
                                extremum_points.append((x_root, y_root, 0))
            except:
                continue
        
        # 标记极值点
        for x_ext, y_ext, second_deriv in extremum_points:
            if second_deriv > 0:
                # 极小值点
                ax.plot(x_ext, y_ext, 'ro', markersize=10, markerfacecolor='red')
                ax.annotate(f'极小值\n({x_ext:.3f}, {y_ext:.3f})', 
                           xy=(x_ext, y_ext), xytext=(10, 10),
                           textcoords='offset points', ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
            elif second_deriv < 0:
                # 极大值点
                ax.plot(x_ext, y_ext, 'go', markersize=10, markerfacecolor='green')
                ax.annotate(f'极大值\n({x_ext:.3f}, {y_ext:.3f})', 
                           xy=(x_ext, y_ext), xytext=(10, -20),
                           textcoords='offset points', ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
            else:
                # 可能是拐点或鞍点
                ax.plot(x_ext, y_ext, 'yo', markersize=10, markerfacecolor='orange')
                ax.annotate(f'驻点\n({x_ext:.3f}, {y_ext:.3f})', 
                           xy=(x_ext, y_ext), xytext=(10, 10),
                           textcoords='offset points', ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
        
        # 在标题中显示极值点数量
        if extremum_points:
            title_extremum = f" - 发现 {len(extremum_points)} 个极值点"
        else:
            title_extremum = " - 未发现极值点"
    else:
        title_extremum = ""
    
    # 8. 添加x轴和y轴
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # 9. 设置图形标题和标签
    equation_str = f"${format_coeff_tex(a)}x^4 {format_coeff_tex(b, True)}x^3 {format_coeff_tex(c, True)}x^2 {format_coeff_tex(d, True)}x {format_coeff_tex(e, True)}$"
    ax.set_title(f"一元四次方程图像: {equation_str}\n区间: [{interval_start}, {interval_end}]{title_extremum}", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    
    # 10. 添加图例
    ax.legend(loc='best', fontsize=10)
    
    # 11. 添加网格
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 12. 设置坐标轴范围
    ax.set_xlim([interval_start, interval_end])
    
    # 自动调整y轴范围以更好地显示图像
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    if y_range == 0:  # 如果函数是常数
        y_range = 1
    ax.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])
    
    # 13. 添加函数值范围信息
    ax.text(0.02, 0.98, f"函数值范围: [{y_min:.3f}, {y_max:.3f}]", 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, ax, extremum_points

def format_coeff(coeff, show_sign=False):
    """格式化系数为字符串"""
    if coeff == 0:
        return ""
    elif coeff == 1:
        return "+ " if show_sign and coeff > 0 else ""
    elif coeff == -1:
        return "- "
    else:
        sign = "+ " if show_sign and coeff > 0 else ""
        return f"{sign}{coeff}"

def format_coeff_tex(coeff, show_sign=False):
    """格式化系数为LaTeX字符串"""
    if coeff == 0:
        return ""
    elif coeff == 1:
        return "+ " if show_sign and coeff > 0 else ""
    elif coeff == -1:
        return "- "
    else:
        sign = "+ " if show_sign and coeff > 0 else ""
        return f"{sign}{coeff}"

def interactive_quartic_plotter():
    """交互式一元四次方程绘图工具"""
    print("=== 一元四次方程绘图工具 ===")
    print("请输入方程的系数 (格式: ax^4 + bx^3 + cx^2 + dx + e)")
    
    try:
        a = float(input("输入系数 a: "))
        b = float(input("输入系数 b: "))
        c = float(input("输入系数 c: "))
        d = float(input("输入系数 d: "))
        e = float(input("输入系数 e: "))
        
        start = float(input("输入区间起点: "))
        end = float(input("输入区间终点: "))
        
        if start >= end:
            print("错误: 区间起点必须小于终点!")
            return
        
        # 绘制图像
        fig, ax, extremum_points = plot_quartic_equation(a, b, c, d, e, start, end)
        
        # 显示极值点信息
        if extremum_points:
            print(f"\n在区间 [{start}, {end}] 内发现的极值点:")
            for i, (x_ext, y_ext, second_deriv) in enumerate(extremum_points):
                if second_deriv > 0:
                    type_str = "极小值点"
                elif second_deriv < 0:
                    type_str = "极大值点"
                else:
                    type_str = "驻点(需进一步分析)"
                print(f"{i+1}. ({x_ext:.4f}, {y_ext:.4f}) - {type_str}")
        
        # 显示图像
        plt.show()
        
    except ValueError:
        print("错误: 请输入有效的数字!")
    except Exception as ex:
        print(f"发生错误: {ex}")

# 示例使用
if __name__ == "__main__":
    # 示例1: f(x) = x^4 - 4x^3 + 6x^2 - 4x + 1
    print("示例1: f(x) = x^4 - 4x^3 + 6x^2 - 4x + 1")
    fig1, ax1, ext1 = plot_quartic_equation(1, -4, 6, -4, 1, -1, 3)
    plt.show()
    
    # 示例2: f(x) = x^4 - 2x^3 - 12x^2
    print("\n示例2: f(x) = x^4 - 2x^3 - 12x^2")
    fig2, ax2, ext2 = plot_quartic_equation(1, -2, -12, 0, 0, -5, 5)
    plt.show()
    
    # 示例3: 更复杂的方程
    print("\n示例3: f(x) = 0.5x^4 - 3x^3 + 2x^2 + 5x - 2")
    fig3, ax3, ext3 = plot_quartic_equation(0.5, -3, 2, 5, -2, -3, 6)
    plt.show()
    
    # 可以取消注释下面的代码以使用交互式工具
    # interactive_quartic_plotter()