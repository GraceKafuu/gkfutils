import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional, Any
import math
from pathlib import Path
from scipy.spatial import KDTree
from main_laser_detect_v2 import *

# 定义数据结构（替代C++的struct）
class PointStruct:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.min_y = 0
        self.max_y = 0
        self.mod_roi_5 = None  # cv2.Mat
        self.meanHSV = None  # cv2.Scalar
        self.kernel5_5 = None  # cv2.Mat

    def calculateNonZeroHeight_RowImage(self, img: cv2.Mat) -> Tuple[int, int]:
        """计算单列非零像素的最小/最大y坐标"""
        non_zero_points = cv2.findNonZero(img)
        if non_zero_points is None or len(non_zero_points) == 0:
            return 10000, 0

        min_y = 10000
        max_y = 0
        for p in non_zero_points:
            y = p[0][1]
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        return min_y, max_y


class SrcImgDataStruct:
    def __init__(self, src_img: cv2.Mat, mask_img: cv2.Mat):
        self.m_src_img = src_img
        self.m_mask_img = self.extractRedRegion(mask_img)
        self.m_line_features = []  # List[PointStruct]
        cv2.imshow("self.m_mask_img", self.m_mask_img)
        # cv2.imwrite(r"G:\Gosion\data\006.Belt_Torn_Det\data\Daba_Data\m_mask_img.png", self.m_mask_img)
        cv2.waitKey(0)
        # 获取每个X坐标下mask的中心点
        for i in range(self.m_mask_img.shape[1]):
            data = PointStruct()
            single_col = self.m_mask_img[:, i:i + 1]  # 取第i列
            nonzero_rows, nonzero_cols = np.nonzero(single_col)
            data.min_y = nonzero_rows[0]
            data.max_y = nonzero_rows[-1]
            data.y = int(np.mean(nonzero_rows))
            data.x = i
            self.m_line_features.append(data)

        # 提取每个点的局部特征
        hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        for t in range(self.m_mask_img.shape[1]):
            i = t
            if t < 2:
                i = 2
            if t > self.m_mask_img.shape[1] - 3:
                i = self.m_mask_img.shape[1] - 3

            # 取前后5个点的min/max y
            data_list = [
                self.m_line_features[i - 2],
                self.m_line_features[i - 1],
                self.m_line_features[i],
                self.m_line_features[i + 1],
                self.m_line_features[i + 2]
            ]
            miny_list = [d.min_y for d in data_list]
            maxy_list = [d.max_y for d in data_list]
            min_h = min(miny_list)
            max_h = max(maxy_list)
            k_h = max_h - min_h

            # 提取ROI
            roi = (i - 2, min_h, 5, k_h)  # x, y, w, h
            kernel5_5 = self.m_mask_img[min_h:min_h + k_h, i - 2:i - 2 + 5].copy()
            # cv2.imshow("kernel5_5", kernel5_5)
            # cv2.waitKey(0)
            # 修正边界点
            if t < 2:
                data = self.m_line_features[t]
            elif t > self.m_mask_img.shape[1] - 3:
                data = self.m_line_features[t]
            else:
                data = self.m_line_features[i]

            data.kernel5_5 = kernel5_5.copy()
            data.mod_roi_5 = hsv_img[min_h:min_h + k_h, i - 2:i - 2 + 5].copy()
            data.meanHSV = cv2.mean(hsv_img[min_h:min_h + k_h, i - 2:i - 2 + 5], mask=kernel5_5)

    def extractRedRegion(self, src: cv2.Mat) -> cv2.Mat:
        """提取红色激光线区域"""
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        # 红色HSV范围（OpenCV中H:0-179, S/V:0-255）
        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 43, 46])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学开运算去噪
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # filtered_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 查找最大轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_idx = 0
        max_w = 0
        max_rect_x = 1000

        for i, cnt in enumerate(contours):
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            if w > max_w:
                max_w = w
                max_idx = i
                max_rect_x = x

        # 验证轮廓是否覆盖整列
        zeroImg = np.zeros_like(src[:, :, 0])
        if max_rect_x != 0 or max_w != src.shape[1]:
            err_msg = f"[ERROR错误]激光线图 标注没到边界（x: {max_rect_x}，宽度: {max_w} ）"
            print(err_msg)
            raise RuntimeError(err_msg)

        cv2.drawContours(zeroImg, contours, max_idx, 255, -1)
        return zeroImg


class FindLine:
    def __init__(self, model_root: str):
        self.m_model_root = model_root
        self.m_templates_struct = []  # List[SrcImgDataStruct]
        self.m_3dImg_h = 100  # 需根据实际需求调整高度
        self.m_3dValImg = None
        self.m_3dHightImg = None
        self.curr_hsv_img = None

        # 读取原图和激光线图
        src_imgs_file = self.getImagesInDir(os.path.join(model_root, "yuantu"))
        mask_imgs_file = self.getImagesInDir(os.path.join(model_root, "jiguangxiantu"))

        if len(src_imgs_file) != len(mask_imgs_file):
            print(f"[ERROR错误]模板图目录：{model_root} 原图和激光线图数量对不上")

        for i in range(len(src_imgs_file)):
            src_name = os.path.basename(src_imgs_file[i]).split('.')[0]
            mask_name = os.path.basename(mask_imgs_file[i]).split('.')[0]

            if src_name != mask_name:
                err_msg = f"[ERROR错误]模板图目录：{model_root} 原图和激光线图数量对不上（原图{len(src_imgs_file)}张，激光线图{len(mask_imgs_file)}张）"
                print(f"[ERROR错误]模板图：{src_imgs_file[i]} 找不到激光线图")
                raise RuntimeError(err_msg)

            src_img = cv2.imread(src_imgs_file[i])
            mask_img = cv2.imread(mask_imgs_file[i])
            self.m_templates_struct.append(SrcImgDataStruct(src_img, mask_img))

        print("----- 读取模板图成功----")
        if len(self.m_templates_struct) > 0:
            h = self.m_templates_struct[0].m_src_img.shape[1]
            self.m_3dValImg = np.zeros((self.m_3dImg_h, h), dtype=np.uint8)
            self.m_3dHightImg = np.zeros((self.m_3dImg_h, h), dtype=np.uint8)

    def getImagesInDir(self, dir_path: str) -> List[str]:
        """获取目录下所有图片路径"""
        if not os.path.exists(dir_path):
            print(f"目录不存在：{dir_path}")
            return []

        img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.svg', '.webp']
        file_list = []

        for file in os.listdir(dir_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in img_ext:
                file_list.append(os.path.join(dir_path, file))

        # 按名称排序
        file_list.sort()
        return file_list

    def getImgPointsFeature(self, img: cv2.Mat, points_num: int, same_th: float, padd: int) -> List[PointStruct]:
        """提取图像点特征"""
        res = []
        if img.shape[1] != len(self.m_templates_struct[0].m_line_features):
            print("----[错误]- 模板图长度和特征图不一样----")
            return res

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.curr_hsv_img = hsv_img.copy()
        showMask = np.zeros_like(img[:, :, 0])

        skip = int((img.shape[1] - 2 * padd) / points_num)
        t = 2
        while t < img.shape[1]:
            i = t
            if t > img.shape[1] - 3:
                i = img.shape[1] - 3
            mod_hsv = self.m_templates_struct[0].m_line_features[i-2].meanHSV
            mask = self.m_templates_struct[0].m_line_features[i-2].kernel5_5
            det_roi = (i-2, 0, 5, img.shape[0])  # x, y, w, h
            search_roi = hsv_img[0:img.shape[0], i-2:i + 3]

            det_mask, h = self.matchFirstSamePointsByHsv(mask, search_roi, mod_hsv, same_th)
            showMask[0:img.shape[0], i-2:i + 3] = det_mask
            # cv2.imshow("showMask",showMask)
            # cv2.waitKey(0)
            data = PointStruct()
            data.x = i-2
            data.y = h
            data.min_y = h
            data.max_y = min(h + mask.shape[0], img.shape[0])

            _roi = (i-2, data.min_y, 5, mask.shape[0])
            data.mod_roi_5 = img[data.min_y:data.min_y + mask.shape[0], i:i + 5].copy()
            data.meanHSV = cv2.mean(hsv_img[data.min_y:data.min_y + mask.shape[0], i-2:i + 3])

            res.append(data)

            if i < padd or i > (img.shape[1] - padd):
                i += 1
            else:
                i += skip
            t += 1

        return res

    def filterOutlierPoints(self, pts: List[Tuple[int, int]], k: int, std_multiplier: float) -> List[Tuple[int, int]]:
        """KDTree过滤离群点"""
        if len(pts) <= k:
            return pts

        # 构建KDTree
        pts_np = np.array(pts, dtype=np.float32)
        # 2. 初始化scipy KDTree（核心：替代OpenCV-FLANN）
        kdtree = KDTree(pts_np)

        # 3. 批量计算所有点的K近邻（效率更高）
        # 返回：距离（已开平方，直接是欧氏距离）、近邻索引
        dists, _ = kdtree.query(pts_np, k=k+1)

        # 3. 计算每个点到K近邻的平均距离（完全复刻C++逻辑）
        avg_distances = []
        for pt in pts:
            # 构造查询点（对齐C++的query_mat）
            query_pt = np.array([pt[0], pt[1]], dtype=np.float32)
            # 搜索K个近邻（C++直接取k个，不含自身；scipy需指定k=k，默认排除自身）
            # 注意：scipy.query(k=k) 返回的是k个近邻（不含自身），距离是欧氏距离（非平方）
            dists, _ = kdtree.query(query_pt, k=k)

            # 对齐C++：累加开平方后的距离（scipy已开平方，直接累加）
            sum_dist = 0.0
            for d in dists:
                sum_dist += d  # C++是sqrt(d)（d是平方），scipy直接是d（欧氏距离）
            # 计算平均距离（对齐C++）
            avg_dist = sum_dist / k
            avg_distances.append(avg_dist)

        # 过滤离群点
        filtered_pts = []
        for i, pt in enumerate(pts):
            if avg_distances[i] < std_multiplier and pt[1] > 10:
                filtered_pts.append(pt)

        return filtered_pts

    def getNearPoints(self, pts: List[Tuple[int, int]], startNum: int, near_num: int, farthest_r: int) -> List[
        Tuple[int, int]]:
        """获取邻近点"""
        out = []
        all_num = len(pts)
        assert 2 * farthest_r <= all_num, "搜索半径大于点数"

        startIdx = startNum - farthest_r
        endIdx = startNum + farthest_r

        if startNum < farthest_r:
            startIdx = 0
            endIdx = 2 * farthest_r
        if (all_num - startNum) < farthest_r:
            startIdx = all_num - 2 * farthest_r
            endIdx = all_num

        sleft_i = startNum
        rleft_i = min(startNum + 1, all_num)

        while True:
            if sleft_i <= startNum and rleft_i >= endIdx:
                break

            if sleft_i > startIdx:
                if pts[sleft_i][1] != 0:
                    out.append(pts[sleft_i])
                    if len(out) > near_num:
                        break

            if rleft_i < endIdx:
                if pts[rleft_i][1] != 0:
                    out.append(pts[rleft_i])
                    if len(out) > near_num:
                        break

            sleft_i -= 1
            rleft_i += 1

        return out

    def smoothPoints(self, pts: List[Tuple[int, int]], out_points: List[Tuple[int, int]], re_size: int) -> bool:
        """平滑点集"""
        if len(pts) < 10:
            return False

        # 初始化out_points
        for pt in pts:
            if int(pt[0]) < len(out_points):
                out_points[int(pt[0])] = pt

        mark_pts = out_points.copy()
        max_h = self.m_templates_struct[0].m_src_img.shape[0]

        for i in range(re_size):
            if out_points[i][0] == i:
                continue

            near_pts = self.getNearPoints(mark_pts, i, 5, 15)
            mean_y = 0.0
            self.calculateYFromX(near_pts, i, mean_y)
            out_points[i] = (i, min(int(mean_y), max_h))

        return True

    def smoothPointsHidth(self, pts: List[Tuple[int, int]], k_size: int) -> List[Tuple[int, int]]:
        """高斯滤波平滑点集y坐标"""
        data_1d = [pt[1] for pt in pts]

        # 转换为OpenCV单列矩阵
        data_2d = np.array(data_1d, dtype=np.float32).reshape(-1, 1)

        # 生成高斯核
        sigma = 1.0
        kernel = cv2.getGaussianKernel(k_size, sigma, cv2.CV_32F)

        # 滤波
        filtered_2d = cv2.filter2D(data_2d, -1, kernel, borderType=cv2.BORDER_REPLICATE)

        # 转换回一维数组
        out_h = filtered_2d.flatten().astype(int)

        # 构造输出点集
        out_pts = []
        for i, pt in enumerate(pts):
            out_pts.append((pt[0], out_h[i]))

        return out_pts

    def addTo3DImg(self, pts: List[Tuple[int, int]], row: int) -> bool:
        """添加点到3D图像（滚动更新：删首行，插末行）"""
        assert len(pts) == self.m_3dValImg.shape[1], "构造3d的点 需要和 m_3dImg 一样"

        val_Img = np.zeros((1, self.m_3dValImg.shape[1]), dtype=np.uint8)
        h_img = np.zeros((1, self.m_3dValImg.shape[1]), dtype=np.uint8)

        tem_val_list = []
        tem_h_list = []

        for i in range(len(pts)):
            mask = self.m_templates_struct[0].m_line_features[i].kernel5_5
            x = pts[i][0]
            y = max(int(pts[i][1] - mask.shape[0] / 2 + 0.5), 0)

            if x > self.m_3dValImg.shape[1] - 5:
                x = self.m_3dValImg.shape[1] - 5
            if y + mask.shape[0] > self.curr_hsv_img.shape[0]:
                y = self.curr_hsv_img.shape[0] - mask.shape[0]

            _roi = (x, y, 5, mask.shape[0])
            hsv_roi = self.curr_hsv_img[y:y + mask.shape[0], x:x + 5]
            hsv_channels = cv2.split(hsv_roi)

            meanV = cv2.mean(hsv_channels[2], mask=mask)
            h = int((y / self.m_templates_struct[0].m_src_img.shape[0]) * 255)

            tem_val_list.append(int(meanV[0]))
            tem_h_list.append(h)

            val_Img[0, i] = int(meanV[0])
            h_img[0, i] = h

        # 移除首行，插入新行到末尾（保持尺寸不变）
        # 处理m_3dValImg
        img_without_first_row = self.m_3dValImg[1:, :].copy()
        self.m_3dValImg = np.vstack([img_without_first_row, val_Img])

        # 处理m_3dHightImg
        img_without_first_row2 = self.m_3dHightImg[1:, :].copy()
        self.m_3dHightImg = np.vstack([img_without_first_row2, h_img])

        return True

    def calculateYFromX(self, pts: List[Tuple[int, int]], x: float, y: float) -> bool:
        """重载：拟合直线并计算y值"""
        k, b = np.nan, np.nan
        if not self.fitLineGetSlope(pts, k, b):
            y = np.nan
            return False

        return self.calculateYFromX_kb(k, b, x, y)

    def fitLineGetSlope(self, pts: List[Tuple[int, int]], k: float, b: float) -> bool:
        """最小二乘法拟合直线，返回斜率和截距"""
        if len(pts) < 2:
            k = np.nan
            b = np.nan
            return False

        # 计算均值
        x_list = [p[0] for p in pts]
        y_list = [p[1] for p in pts]
        x_mean = np.mean(x_list)
        y_mean = np.mean(y_list)

        # 计算分子分母
        numerator = 0.0
        denominator = 0.0
        for x, y in zip(x_list, y_list):
            dx = x - x_mean
            dy = y - y_mean
            numerator += dx * dy
            denominator += dx * dx

        # 处理垂直/水平直线
        if abs(denominator) < 1e-8:
            k = np.inf
            b = np.nan
        else:
            k = numerator / denominator
            b = y_mean - k * x_mean

        return True

    def calculateYFromX_kb(self, k: float, b: float, x: float, y: float) -> bool:
        """根据斜率和截距计算y值"""
        if np.isnan(k):
            # print("错误：斜率无效（未拟合直线）！")
            y = 0
            return False

        if np.isinf(k):
            # print("错误：垂直直线（x固定），无法根据x计算y！")
            y = 0
            return False

        y = k * x + b
        return True

    def maskTemplateMatch(self, src: cv2.Mat, templ: cv2.Mat, mask: cv2.Mat, match_method: int) -> Tuple[
        cv2.Mat, Tuple[int, int]]:
        """带掩码的模板匹配"""
        result = cv2.matchTemplate(src, templ, match_method, mask=mask)

        # 查找最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_loc = min_loc
        else:
            best_loc = max_loc

        return result, best_loc

    # 全局函数
    def matchByMask(self, mask: cv2.Mat, src_mask: cv2.Mat, threshold: float) -> int:
        """通过掩码匹配"""
        first_match = 0
        mid_h = int(mask.shape[0] / 2)
        k_h = src_mask.shape[0] - mask.shape[0]
        max_score = 0.0
        first_idx = 0
        first_search = False

        for i in range(k_h):
            roi = (0, i, 5, mask.shape[0])
            if cv2.countNonZero(src_mask[i:i + mask.shape[0], 0:5]) == 0:# 加速
                continue
            mean = cv2.mean(src_mask[i:i + mask.shape[0], 0:5], mask=mask)
            _score = mean[0] / 255

            if first_search:
                diff = i - first_idx
                if diff < 5:
                    if _score > max_score:
                        max_score = _score
                        first_match = i
                else:
                    break

            if _score > threshold:
                max_score = _score
                first_search = True
                first_idx = i
                first_match = i

        return first_match

    def matchFirstSamePointsByHsv(self, mask: cv2.Mat, src_hsv: cv2.Mat, meanHsv: Tuple[float, float, float],
                                  threshold: float) -> Tuple[cv2.Mat, int]:
        """基于HSV匹配第一个相同点"""
        assert mask.shape[1] == src_hsv.shape[1], "matchFirstSamePointsByHsv 两者宽度不匹配"

        # 计算HSV阈值范围
        lower = np.array([
            int(meanHsv[0] * 0.5),
            int(meanHsv[1] * 0.5),
            int(meanHsv[2] * 0.8)
        ], dtype=np.uint8)

        upper = np.array([
            min(int(meanHsv[0] * 2), 255),
            min(int(meanHsv[1] * 2), 255),
            min(int(meanHsv[2] * 1.5), 255)
        ], dtype=np.uint8)

        # 生成掩码
        src_mask = cv2.inRange(src_hsv, lower, upper)

        # 膨胀mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bigMask = cv2.dilate(mask, kernel)
        # cv2.imshow("mask", mask)
        # cv2.imshow("src_mask", src_mask)
        # cv2.waitKey(1)
        first_match = 0
        _mask = mask.copy()
        while True:
            first_match = self.matchByMask(_mask, src_mask, threshold)
            if first_match == 0:
                threshold -= 0.1
                _mask = bigMask.copy()
            if threshold < 0.3 or first_match != 0:
                break

        return src_mask, first_match



# 辅助函数：获取点集极值
def getPointsMinMax(pts: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """获取点集的min/max x/y"""
    if not pts:
        return np.inf, -np.inf, np.inf, -np.inf

    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)

    return min_x, max_x, min_y, max_y


def main1():
    # 测试示例
    import datetime
    # 初始化FindLine类
    try:
        m_findline = FindLine(r"G:/Gosion/data/006.Belt_Torn_Det/data/Daba_Data/mugun_gaoliangfanguang2/")  # 替换为实际模板目录
        print("FindLine初始化成功")
        cap = cv2.VideoCapture(r'G:\Gosion\data\006.Belt_Torn_Det\data\Daba_Data\mugun_gaoliangfanguang.avi')
        if not cap.isOpened():
            print("无法打开视频文件")
            exit()

        while True:
            # 读取一帧视频
            ret, frame = cap.read()

            if not ret:
                print("无法读取帧，退出循环")
                break
            # 测试读取图像并提取特征
            start_time = datetime.datetime.now()  # 计时开始

            # 克隆帧用于绘制
            showImg = frame.copy()
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # 提取线特征点
            line_datas = m_findline.getImgPointsFeature(frame, 400, 0.2, 100)
            elapsed_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
            print(f"耗时 getImgPointsFeature ：{elapsed_ms} ms")
            # 绘制原始特征点（蓝色）
            for data in line_datas:
                x = int(data.x)
                y = int(data.y)
                cv2.circle(showImg, (x, y), 2, (255, 0, 0), -1, cv2.LINE_AA)

            # 过滤离群点
            pts = []
            for data in line_datas:
                pts.append((int(data.x), int(data.y)))  # 转换为 (x,y) 元组
            filter_pts = m_findline.filterOutlierPoints(pts, 5, 8.0)
            # for (x, y) in filter_pts:# 绘制过滤后的点（红色）
            #     cv2.circle(showImg, (x, y), 2, (0, 0, 200), -1, cv2.LINE_AA)

            # 对保留的点进行插值补全
            out_pts = [(0, 0)] * frame.shape[1]  # 初始化输出容器，长度=帧的列数
            flag = m_findline.smoothPoints(filter_pts, out_pts, frame.shape[1])
            # smooth_pts = m_findline.smoothPointsHidth(out_pts, k_size=5)  # k_size对应原代码的核大小，默认5。。这里本来想平滑一下Y坐标的，
            #
            # # 绘制平滑后的点（洋红色）
            for (x, y) in out_pts:
                cv2.circle(showImg, (int(x), int(y)), 2, (255, 0, 255), -1, cv2.LINE_AA)

            for (x, y) in out_pts:
                cv2.circle(mask, (int(x), int(y)), 5, 255, -1, cv2.LINE_AA)

            #
            # # 添加到3D高度图
            m_findline.addTo3DImg(out_pts, 100)

            # 计算耗时（毫秒）
            elapsed_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
            print(f"耗时：{elapsed_ms} ms")

            # 显示图像
            cv2.imshow("hit", m_findline.m_3dHightImg)
            cv2.imshow("valu", m_findline.m_3dValImg)
            cv2.imshow("frame", frame)
            cv2.imshow("showImg", showImg)
            cv2.imshow("mask", mask)

            # 等待按键（0为无限等待，按任意键继续；若需实时显示，改为 cv2.waitKey(1)）
            cv2.waitKey(1)
    except Exception as e:
        print(f"错误：{e}")


def main2():
    # 测试示例
    import datetime
    # 初始化FindLine类
    try:
        m_findline = FindLine(r"G:/Gosion/data/006.Belt_Torn_Det/data/Daba_Data/mugun_gaoliangfanguang/")  # 替换为实际模板目录
        print("FindLine初始化成功")
        cap = cv2.VideoCapture(r'G:\Gosion\data\006.Belt_Torn_Det\data\Daba_Data\mugun_gaoliangfanguang.avi')
        if not cap.isOpened():
            print("无法打开视频文件")
            exit()

        # data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\cls3\v3\train\Random_Selected\test"
        # save_path = data_path + "_result"
        # os.makedirs(save_path, exist_ok=True)

        # file_list = os.listdir(data_path)


        # for f in file_list:
        while True:
            # 读取一帧视频
            ret, frame = cap.read()

            if not ret:
                print("无法读取帧，退出循环")
                break
            # 测试读取图像并提取特征
            start_time = datetime.datetime.now()  # 计时开始

            # f_abs_path = os.path.join(data_path, f)
            # fname = os.path.splitext(f)[0]
            # frame = cv2.imread(f_abs_path)
            # frame = cv2.resize(frame, (1600, 548))

            # 克隆帧用于绘制
            showImg = frame.copy()
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # 提取线特征点
            line_datas = m_findline.getImgPointsFeature(frame, 400, 0.3, 100)
            elapsed_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
            print(f"耗时 getImgPointsFeature ：{elapsed_ms} ms")
            # 绘制原始特征点（蓝色）
            for data in line_datas:
                x = int(data.x)
                y = int(data.y)
                cv2.circle(showImg, (x, y), 2, (255, 0, 0), -1, cv2.LINE_AA)

            # 过滤离群点
            pts = []
            for data in line_datas:
                pts.append((int(data.x), int(data.y)))  # 转换为 (x,y) 元组
            filter_pts = m_findline.filterOutlierPoints(pts, 5, 8.0)
            # for (x, y) in filter_pts:# 绘制过滤后的点（红色）
            #     cv2.circle(showImg, (x, y), 2, (0, 0, 200), -1, cv2.LINE_AA)

            # 对保留的点进行插值补全
            out_pts = [(0, 0)] * frame.shape[1]  # 初始化输出容器，长度=帧的列数
            flag = m_findline.smoothPoints(filter_pts, out_pts, frame.shape[1])
            # smooth_pts = m_findline.smoothPointsHidth(out_pts, k_size=5)  # k_size对应原代码的核大小，默认5。。这里本来想平滑一下Y坐标的，
            #
            # # 绘制平滑后的点（洋红色）
            for (x, y) in out_pts:
                cv2.circle(showImg, (int(x), int(y)), 2, (255, 0, 255), -1, cv2.LINE_AA)

            points = []
            for (x, y) in out_pts:
                if y < 10: 
                    continue
                points.append([int(x), int(y)])
                cv2.circle(mask, (int(x), int(y)), 5, 255, -1, cv2.LINE_AA)

            # =========================
            img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\Daba_Data\m_mask_img.png"
            img_test = cv2.imread(img_path, 0)
            points_test = np.where(img_test > 0)

            points_test[0][(points_test[0] > 50) & (points_test[0] < 100)] = 0
            fit_model = np.polyfit(points_test[1], points_test[0], deg=4)
            img_test_bgr = cv2.cvtColor(img_test, cv2.COLOR_GRAY2BGR)
            plot_out_test = fit_plot(img_test_bgr, fit_model, r=4)

            cv2.imshow("plot_out_test", plot_out_test)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
 

            imgsz = mask.shape
            width = int(imgsz[1] / 4)
            points = np.array(points)
            for idx in range(4):
                if idx < 3:
                    imgi = mask[:, idx * width : idx * width + width]
                    # pointsi = points[idx * width : idx * width + width]
                    # pointsi = points[idx * width : idx * width + width, :]
                    t_points = points[:, 0] - idx * width
                    t_points_y = points[:, 1]

                    pointsi_x = t_points[ idx * width : idx * width + width]
                    pointsi_y = t_points_y[ idx * width : idx * width + width]


                else:
                    imgi = mask[:, idx * width :]
                    # t_points = points[:, 0] - idx * width
                    # pointsi = t_points[idx * width :]
                    t_points = points[:, 0] - idx * width
                    t_points_y = points[:, 1]

                    pointsi_x = t_points[ idx * width :]
                    pointsi_y = t_points_y[ idx * width :]

                # points = np.array(points)
                # points = out_pts
                # fit_model = np.polyfit(pointsi[:, 0], pointsi[:, 1], deg=4)
                fit_model = np.polyfit(pointsi_x, pointsi_y, deg=2)
                # print("fname: {} fit_model: {}".format(fname, fit_model))
                plot_out = fit_plot(imgi, fit_model, r=4)

                cv2.imshow("plot_out", plot_out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            #
            # # 添加到3D高度图
            # m_findline.addTo3DImg(out_pts, 100)

            # 计算耗时（毫秒）
            elapsed_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000
            print(f"耗时：{elapsed_ms} ms")

            # 显示图像
            cv2.imshow("hit", m_findline.m_3dHightImg)
            cv2.imshow("valu", m_findline.m_3dValImg)
            cv2.imshow("frame", frame)
            cv2.imshow("showImg", showImg)
            cv2.imshow("mask", mask)
            cv2.imshow("plot_out", plot_out)

            # 等待按键（0为无限等待，按任意键继续；若需实时显示，改为 cv2.waitKey(1)）
            cv2.waitKey(0)
    except Exception as e:
        print(f"错误：{e}")


# 测试示例
if __name__ == "__main__":
    # main1()
    main2()