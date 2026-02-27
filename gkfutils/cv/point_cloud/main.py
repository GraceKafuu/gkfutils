import os
import cv2
import copy
import numpy as np
import open3d as o3d


"""
不同点云数据格式转换: https://zhuanlan.zhihu.com/p/705630816
"""

def main_icp_test():
    # Ref: https://blog.csdn.net/qq_36592837/article/details/149982354

    source = o3d.io.read_point_cloud(r"G:\Gosion\code\Python\PointCloud\data\36abb-main\bunny.pcd")
    
    # 原始点云x轴平移0.1，y轴平移0.15，z轴平移0.2，然后加上噪声，最后新的点云作为目标点云
    # 平移
    target = copy.deepcopy(source)
    points = np.asarray(target.points)
    t = np.array([0.1, 0.15, 0.2])
    target = target.translate(t, relative=True)  # relative=True 表示“增量”平移
    
    # 加噪声
    mu, sigma = 0, 0.001  # 均值和标准差根据密度调节
    random_numbers = np.random.normal(mu, sigma, size=(points.shape[0], 3))
    points += random_numbers
    target.points = o3d.utility.Vector3dVector(points)
    
    target.paint_uniform_color([0, 1, 0])  # 给目标点云赋予单色
    
    # --------------------计算法向量---------------------
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    # 复制点云，防止更改
    source1 = copy.deepcopy(source)
    target1 = copy.deepcopy(target)
    
    # 配置参数
    threshold = 1  # 配准点对的最大配准距离阈值
    trans_init = np.eye(4)  # 这里是初始变换矩阵，一般由粗配准提供，这里使用单位矩阵代替
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print("配准前信息：", evaluation)  # 这里输出的是初始位置的fitness和RMSE，fitness是覆盖率，代表对应点之间的覆盖程度，RMSE是均方根误差，也是对应点之间的
    
    icp_p2plane = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),    # 执行点对面的ICP算法
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))  # 设置最大迭代次数
    print("配准后信息：", icp_p2plane)  # 输出ICP相关信息
    print("两块点云之间的配准矩阵:", icp_p2plane.transformation)
    
    source2 = copy.deepcopy(source1)
    source2.transform(icp_p2plane.transformation)
    
    o3d.visualization.draw_geometries([source1, target1],
                                    window_name="两点云初始位置",
                                    width=1200, height=800,
                                    left=50, top=50)
    o3d.visualization.draw_geometries([source2, target1],
                                    window_name="配准后两点云初始位置",
                                    width=1200, height=800,
                                    left=50, top=50)


if __name__ == '__main__':
    main_icp_test()

