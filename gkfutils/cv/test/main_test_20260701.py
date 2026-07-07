# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. 读取图像并转换为灰度图
# # 提示：你可以使用任意包含几何形状或建筑的图片
# # img = cv2.imread('chess_board.jpg') 

# img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\rename_20250807_0012682.jpg"
# # img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\Video_2025_08_27_173442_1_output_000001197.jpg"
# img = cv2.imread(img_path) 
# if img is None:
#     # 如果没有图片，我们用numpy创建一个简单的测试图（黑白方块）
#     img = np.zeros((400, 400, 3), dtype=np.uint8)
#     cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
#     # 添加一个低对比度的灰色方块，用来模拟“不明显”的特征
#     cv2.rectangle(img, (150, 150), (250, 250), (180, 180, 180), -1)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# # 2. 执行 Harris 角点检测
# # 参数：邻域大小=2, 索贝尔算子孔径=3, Harris检测器自由参数=0.04
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# # 膨胀结果，仅为了方便在可视化时放大角点标记
# dst = cv2.dilate(dst, None)

# # 3. 通过不同的阈值区分明显与不明显特征
# img_strong = img.copy()
# img_weak = img.copy()

# # 明显特征：阈值设得较高（例如最大响应值的 5%）
# threshold_strong = 0.05 * dst.max()
# img_strong[dst > threshold_strong] = [0, 0, 255] # 用红色标记

# # 不明显特征：阈值设得较低（例如最大响应值的 0.5% 到 5% 之间）
# threshold_weak_lower = 0.005 * dst.max()
# threshold_weak_upper = 0.05 * dst.max()
# img_weak[(dst > threshold_weak_lower) & (dst <= threshold_weak_upper)] = [0, 255, 255] # 用绿色标记

# # 4. 结果可视化
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.title("Distinct Features (Red)")
# plt.imshow(cv2.cvtColor(img_strong, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Indistinct Features (Green)")
# plt.imshow(cv2.cvtColor(img_weak, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.show()



# import cv2
# import matplotlib.pyplot as plt

# # 1. 读取图像并转换为灰度图
# # img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\rename_20250807_0012682.jpg"
# img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\Video_2025_08_27_173442_1_output_000001197.jpg"
# img = cv2.imread(img_path) 
# if img is None:
#     # 如果没有准备图片，这里创建一个随机纹理图做演示
#     import numpy as np
#     img = np.uint8(np.random.rand(400, 400, 3) * 255)
#     cv2.rectangle(img, (100, 100), (300, 300), (120, 200, 100), -1)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # ==========================================
# # 方法一：SIFT 算法（最稳定、最精准）
# # ==========================================
# # 创建 SIFT 对象
# sift = cv2.SIFT_create()
# # 检测特征点并计算描述子
# kp_sift, des_sift = sift.detectAndCompute(gray, None)
# # 绘制特征点（RichKeypoints 会绘制出特征点的大小和方向）
# img_sift = cv2.drawKeypoints(img, kp_sift, None, 
#                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # ==========================================
# # 方法二：ORB 算法（实时性极佳、效率高）
# # ==========================================
# # 创建 ORB 对象（可以通过 nfeatures 限制最大特征点数量）
# orb = cv2.ORB_create(nfeatures=500)
# # 检测特征点并计算描述子
# kp_orb, des_orb = orb.detectAndCompute(gray, None)
# # 绘制特征点
# img_orb = cv2.drawKeypoints(img, kp_orb, None, 
#                             color=(0, 255, 0), flags=0)

# # ==========================================
# # 结果可视化
# # ==========================================
# plt.figure(figsize=(14, 7))

# plt.subplot(1, 2, 1)
# plt.title(f"SIFT (Stable) - Found {len(kp_sift)} points")
# plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title(f"ORB (Fast) - Found {len(kp_orb)} points")
# plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.show()




import cv2
import numpy as np

def detect_laser_cracks_stable(image_path, jump_threshold=18, min_gap_width=5):
    """
    鲁棒检测绿色激光线条的断裂和阶跃错位（真实裂纹），
    使用局部自适应增强(CLAHE)、脊线追踪和连接器算法，
    能有效应对噪声、反光和弱信号。
    :param image_path: 图片路径
    :param jump_threshold: 纵向错位阈值（单位：像素），小于此值视为正常弧线，大于视为错位
    :param min_gap_width: 两处断裂判定为独立断裂的最小间隔，也用于聚类
    :return: 标注后的图片, 断裂点的X坐标列表
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 无法加载图片 {image_path}")
        return None, []
    
    h, w, _ = img.shape
    
    # --- 1. 预处理：降噪与局部增强 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_channel = img[:, :, 1] # 提取G通道
    
    # 降噪，保护激光脊线细节
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    g_channel = cv2.GaussianBlur(g_channel, (5, 5), 0)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # 这个核心步骤显著增强暗部激光信号，同时限制对背景噪声的过度放大和压制反光
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced_gray = clahe.apply(gray)
    enhanced_g = clahe.apply(g_channel)

    h_e, w_e = enhanced_gray.shape
    y_coords = np.full(w, np.nan)
    g_scores = np.full(w, np.nan) # 用于记录追踪到的原图G信号强度

    # --- 2. 脊线特征提取（不仅是质心） ---
    # 对增强后的图像在纵向上寻找局部极大值（最亮点），代表激光脊线
    for x in range(w):
        pts = np.where(enhanced_gray[:, x] > 20)[0] # 设置较低的强度阈值，因为CLAHE已经增强了信号
        if len(pts) > 0:
            # 寻找每一列的最亮簇的中心，对增强后的图像鲁棒
            # 这里我们假设增强后的图像脊线是最亮的
            max_pts = np.where(enhanced_gray[:, x] == np.max(enhanced_gray[pts, x]))[0]
            y_coords[x] = np.mean(max_pts)
            g_scores[x] = np.max(g_channel[:, x]) # 记录增强前原图G通道最大值
            
    # --- 3. 鲁棒线追踪（解决暂时信号丢失和噪声伪断裂） ---
    # 使用连接器逻辑，如果距离小于阈值，就将相邻列连接，飞跃噪声区域
    smoothed_coords = np.copy(y_coords)
    tracked_g_scores = np.copy(g_scores)
    
    missing_counter = 0
    max_missing_columns = 12 # 连续缺失多列才判定为信号彻底丢失
    last_known_y = np.nan
    
    # 可选：可以应用一个小的平滑核以去除像素级纵向抖动，同时保留阶跃
    # kernel = np.array([-1, 2, -1]) # 对垂直方向应用，但在列方向遍历中不适用
    # 这里保持简单的插值逻辑，更复杂应用可以加入卡尔曼滤波
    
    for x in range(w):
        if not np.isnan(y_coords[x]):
            missing_counter = 0
            last_known_y = y_coords[x]
        else:
            missing_counter += 1
            if missing_counter <= max_missing_columns and not np.isnan(last_known_y):
                # 如果丢失信号短，则保持最后已知值（飞跃噪声）
                smoothed_coords[x] = last_known_y
                tracked_g_scores[x] = 0 # 将该处G信号原值标记为0，表示信号弱但被连接

    # --- 4. 裂纹检测（多判据，包括阶跃和信号突变） ---
    cracks_x = []
    
    # 可选：还可以提取每一列的最暗像素作为另一个特征
    
    for x in range(1, w):
        prev_y = smoothed_coords[x-1]
        curr_y = smoothed_coords[x]
        
        if not np.isnan(prev_y) and not np.isnan(curr_y):
            # 判据1：垂直跳变
            dy = abs(curr_y - prev_y)
            
            # 判据2：亮度突变特性（对错位裂纹非常有效）
            # 分析追踪到的原图G信号强度差值
            # 注意：在“飞跃”区， tracked_g_scores 被置为0，产生dg极大
            g_prev = tracked_g_scores[x-1]
            g_curr = tracked_g_scores[x]
            
            dg = abs(g_curr - g_prev)
            
            # 裂纹判定逻辑：
            # A. 显著垂直阶跃
            # B. 中等跳变（如由于裂纹错位引起局部亮后灭）同时伴随G信号亮后突灭特性
            # C. 我们之前连接噪声的“飞跃区”产生了较大的dg，如果追踪正常，这里忽略
            
            # 更好的dg判别，避免对噪声飞跃区的误判
            # 这里设定一个亮度阈值，只有信号足够亮后的突灭才有效
            g_bright_thresh = 150 # G通道增强前灰度值
            
            is_crack = (dy > jump_threshold) or (dy > 12 and dg > g_bright_thresh)
            
            if is_crack:
                # 记录裂纹位置
                cracks_x.append(x)

    # --- 5. 聚类：将连续的、临近的检测合并为一个点 ---
    refined_breaks = []
    if len(cracks_x) > 0:
        cracks_x = sorted(cracks_x)
        current_cluster = [cracks_x[0]]
        for x in cracks_x[1:]:
            if x - current_cluster[-1] <= min_gap_width:
                current_cluster.append(x)
            else:
                refined_breaks.append(int(np.mean(current_cluster)))
                current_cluster = [x]
        refined_breaks.append(int(np.mean(current_cluster)))

    # --- 6. 标注结果 ---
    # 这里标注一个详细的信息，不仅画线，还标出裂纹类型（如果有，通过更复杂的判据）
    annotated_img = img.copy()
    
    for bx in refined_breaks:
        # 画一条红色的垂直线标注断裂位置
        cv2.line(annotated_img, (bx, 0), (bx, h), (0, 0, 255), 2)
        # 在上方标出文本
        cv2.putText(annotated_img, f"Crack X:{bx}", (bx + 5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    return annotated_img, refined_breaks

# --- 使用示例 ---
if __name__ == "__main__":
    # 使用一个更复杂的图像作为示例，比如图片5，有反光和真实裂纹
    img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\rename_20250807_0012682.jpg"
    # img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\Video_2025_08_27_173442_1_output_000001197.jpg"
    img = cv2.imread(img_path) 
    
    # 运行检测，适当微调阈值以获得最佳稳定性
    # 对于高稳定性的要求，jump_threshold 应设定为一个能极好排除弧线本身弯曲的值
    annotated_img, crack_points = detect_laser_cracks_stable(img_path, jump_threshold=18)
    
    print(f"成功检测到 {len(crack_points)} 处潜在裂纹，横坐标位置分别为: {crack_points}")
    
    # 显示对比图：原图 vs 处理后标注图
    # 这里我们生成一张对比图
    img_orig = cv2.imread(img_path)
    concatenated_img = np.hstack((img_orig, annotated_img))
    
    # cv2.imwrite("stable_detection_comparison.jpg", concatenated_img)
    cv2.imshow("Original vs Detection", annotated_img)
    cv2.waitKey(0)