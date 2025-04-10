import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import tqdm
import cv2
import time
import os


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches, warped_corners


def perspective_transform(img, rect):
    """
    透视变换
    tl, tr, br, bl = rect
    """
    tl, tr, br, bl = rect
    # tl, tr, br, bl = np.array([tl[0] - 20, tl[1] - 20]), np.array([tr[0] + 20, tr[1] - 20]), np.array([br[0] + 20, br[1] + 20]), np.array([bl[0] - 20, bl[1] + 20])
    # rect_new = np.array([tl[0] - 20, tl[1] - 20]), np.array([tr[0] + 20, tr[1] - 20]), np.array([br[0] + 20, br[1] + 20]), np.array([bl[0] - 20, bl[1] + 20])
    # 计算宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 计算高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 定义变换后新图像的尺寸
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1],
                   [0, maxHeight-1]], dtype='float32')
    # 变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def connected_components_analysis(img, connectivity=8, area_thr=100, h_thr=8, w_thr=8):
    """
    stats: [x, y, w, h, area]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    
    areas = stats[:, -1]  # stats[:, cv2.CC_STAT_AREA]
    for i in range(1, num_labels):
        if areas[i] < area_thr or stats[i, 3] > h_thr:
            labels[labels == i] = 0

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 256)
        output[:, :, 1][mask] = np.random.randint(0, 256)
        output[:, :, 2][mask] = np.random.randint(0, 256)

    return num_labels, labels, stats, centroids, output


def z_score(x, mean, std):
    return (x - mean) / std


def extract_in_range(arr, low, high):
    """提取数组中在[low, high]范围内的元素"""
    mask = (arr >= low) & (arr <= high)
    return arr[mask]


def extract_specific_color(img, lower, upper, color="green"):
    """
    提取图中指定颜色区域
    :param img: 输入图像
    :param lower: 颜色下限
    :param upper: 颜色上限
    :param color: 颜色名称
    :return: mask, result, binary, binary_otsu
    
    https://news.sohu.com/a/569474695_120265289
    https://blog.csdn.net/yuan2019035055/article/details/140495066
    """
    assert len(img.shape) == 3, "len(img.shape) != 3"

    if color == "black":
        lower = (0, 0, 0)
        upper = (180, 255, 46)
    elif color == "gray":
        lower = (0, 0, 46)
        upper = (180, 43, 220)
    elif color == "white":
        lower = (0, 0, 221)
        upper = (180, 30, 255)
    elif color == "red":
        lower0 = (0, 43, 46)
        upper0 = (10, 255, 255)
        lower1 = (156, 43, 46)
        upper1 = (180, 255, 255)
    elif color == "orange":
        lower = (11, 43, 46)
        upper = (25, 255, 255)
    elif color == "yellow":
        lower = (26, 43, 46)
        upper = (34, 255, 255)
    elif color == "green":
        lower = (35, 43, 46)
        upper = (77, 255, 255)
    elif color == "cyan":
        lower = (78, 43, 46)
        upper = (99, 255, 255)
    elif color == "blue":
        lower = (100, 43, 46)
        upper = (124, 255, 255)
    elif color == "purple":
        lower = (125, 43, 46)
        upper = (155, 255, 255)
    else:
        assert lower is not None and upper is not None and color not in ['black', 'gray', 'white', 'red', 'orange', 'yellow','green', 'cyan', 'blue', 'purple'], "Please choose color \
        from ['black', 'gray', 'white', 'red', 'orange', 'yellow','green', 'cyan', 'blue', 'purple']. If not in the list, please input the 'lower' and 'upper' HSV value of the color."
        
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == "red":
        mask0 = cv2.inRange(hsv_img, lower0, upper0)
        mask1 = cv2.inRange(hsv_img, lower1, upper1)
        mask = cv2.bitwise_or(mask0, mask1)
    else:
        mask = cv2.inRange(hsv_img, lower, upper)

    # 可视化结果（可选）
    # 将掩码应用到原图上，显示提取的颜色区域
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, result, binary, binary_otsu


def detect_oil_level_by_edge(img, reduce_sum_thr=100, y_range=None, color=(255, 0, 255), thickness=4):
    """
    通过一些边缘信息来检测油位
    y_range = (115, 325)
    """
    imgsz = img.shape[:2]
    # CLAHE增强
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 横线检测
    k1 = np.array([
        [-1, -1, -1], 
        [0, 0, 0],
        [1, 1, 1]
    ])
    k1_filtered = cv2.filter2D(binary, -1, k1)

    # 连通域分析去除一些小点
    num_labels, labels, stats, centroids, output = connected_components_analysis(k1_filtered, connectivity=8, area_thr=25, h_thr=20, w_thr=50)
    _, labels_binary = cv2.threshold(np.uint8(labels), 0, 255, cv2.THRESH_BINARY)

    # 
    row_sums = np.sum(labels_binary, axis=1)

    # # if row_sums.size > 0:
    # mean_value = np.mean(row_sums)
    # median_value = np.median(row_sums)
    # std_value = np.std(row_sums)
    # zscore = z_score(row_sums, median_value, std_value)

    coords1 = np.where(row_sums > reduce_sum_thr)
    # coords2 = np.where(abs(zscore) > zScoreThr)

    if y_range is not None:
        oil_level = extract_in_range(coords1[0], y_range[0], y_range[1])
        if len(oil_level) == 0:
            oil_level = None
            oil_level_percent = None
        else:
            oil_level = np.mean(oil_level)
    else:
        if coords1[0].size == 0:
            oil_level = None
            oil_level_percent = None
        else:
            oil_level = np.mean(coords1)
    
    if oil_level is not None:
        cv2.line(img, (0, round(oil_level)), (imgsz[1], round(oil_level)), color, thickness)
        oil_level_percent = round((y_range[1] - oil_level) / (y_range[1] - y_range[0]) * 100, 2)

    return img, oil_level, oil_level_percent


def detect_oil_level_by_color(img, lower=None, upper=None, object_color=["yellow", "orange"], reduce_sum_thr=100, area_thresh=100, y_range=None, color=(255, 255, 0), thickness=4):
    imgsz = img.shape[:2]

    color_bin_otsus = []
    for oc in object_color:
        mask, color_result, color_bin, color_bin_otsu = extract_specific_color(img, lower=None, upper=None, color=oc)
        color_bin_otsus.append(color_bin_otsu)

    color_bin_otsu_new = np.zeros(imgsz, dtype=np.uint8)
    for i in range(len(color_bin_otsus)):
        color_bin_otsu_new = cv2.bitwise_or(color_bin_otsu_new, color_bin_otsus[i])
    
    contours, hierarchy = cv2.findContours(color_bin_otsu_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    specific_color_y = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < area_thresh: continue
        specific_color_y.append(y)

    if len(specific_color_y) == 0:
        return img, None, None
    
    oil_level = np.min(specific_color_y)
    cv2.line(img, (0, round(oil_level)), (imgsz[1], round(oil_level)), color, thickness)
    oil_level_percent = round((y_range[1] - oil_level) / (y_range[1] - y_range[0]) * 100, 2)
    
    return img, oil_level, oil_level_percent
    

def check_warped_corners(warped_corners, imgsz):
    warped_corners_x = warped_corners[:, :, 0]
    warped_corners_y = warped_corners[:, :, 1]
    for i in warped_corners_x:
        if i < 0 or i > imgsz[1]:
            return False
    for i in warped_corners_y:
        if i < 0 or i > imgsz[0]:
            return False
        
    return True


def check_range(oil_range, imgsz):
    n1 = 0
    n2 = 0
    for i in oil_range:
        if i >= 0 and i <= 1:
            n1 += 1
        if i >= 0 and i <= 100:
            n2 += 1

    if n1 == 2 and n2 < 2:
        oil_range = (round(oil_range[0] * imgsz[0]), round(oil_range[1] * imgsz[0]))
    elif n1 < 2 and n2 == 2:
        oil_range = (round(oil_range[0] / 100 * imgsz[0]), round(oil_range[1] / 100 * imgsz[0]))
    elif n1 == 2 and n2 == 2:
        oil_range = (round(oil_range[0] * imgsz[0]), round(oil_range[1] * imgsz[0]))
    else:
        return oil_range
    
    return oil_range
    


def detect_oil_level_by_xfeat_onnx(ort_session, im1, im2, alg=1, oil_range=(115, 325), reduce_sum_thr=500, area_thresh=100, color=(255, 255, 0), thickness=4):
    """
    油位计检测：
    alg 0: 通过边缘检测和连通域分析计算液面的位置
    alg 1: 通过颜色分析计算液面的位置
    alg 2: 综合0和1
    alg 3: 其他方法(备用, 例如深度学习算法)
    """
    assert alg in [0, 1, 2, 3], "alg should be in [0, 1, 2, 3]!"

    input_array_1 = im1.transpose(2, 0, 1).astype(np.float32)
    input_array_1 = np.expand_dims(input_array_1, axis=0)
    input_array_2 = im2.transpose(2, 0, 1).astype(np.float32)
    input_array_2 = np.expand_dims(input_array_2, axis=0)

    inputs = {
        ort_session.get_inputs()[0].name: input_array_1,
        ort_session.get_inputs()[1].name: input_array_2
    }

    outputs = ort_session.run(None, inputs)

    matches = outputs[0]
    batch_indexes = outputs[1]
    mkpts_0, mkpts_1 = matches[batch_indexes == 0][..., :2], matches[batch_indexes == 0][..., 2:]


    canvas, warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)

    flag = check_warped_corners(warped_corners, im2.shape[:2])
    if not flag:
        return im2, None, None

    warped = perspective_transform(im2, np.array(warped_corners, dtype='float32').reshape(-1, 2))
    warpedsz = warped.shape[:2]
    warped_cp = warped.copy()

    # 判断量程
    oil_range = check_range(oil_range, warpedsz)
    

    if alg == 0:
        detected_img, oil_level, oil_level_percent = detect_oil_level_by_edge(warped, reduce_sum_thr=reduce_sum_thr, y_range=oil_range, color=color, thickness=thickness)
        if oil_level_percent < 0 or oil_level_percent > 100:
            oil_level_percent = None
        return detected_img, oil_level, oil_level_percent
    elif alg == 1:
        detected_img2, oil_level2, oil_level_percent2 = detect_oil_level_by_color(
            warped_cp, lower=None, upper=None, object_color=["yellow", "orange"], reduce_sum_thr=reduce_sum_thr, area_thresh=area_thresh, y_range=oil_range, color=color, thickness=thickness
        )
        if oil_level_percent2 < 0 or oil_level_percent2 > 100:
            oil_level_percent2 = None
        return detected_img2, oil_level2, oil_level_percent2
    elif alg == 2:
        detected_img, oil_level, oil_level_percent = detect_oil_level_by_edge(warped, reduce_sum_thr=reduce_sum_thr, y_range=oil_range, color=color, thickness=thickness)
        detected_img2, oil_level2, oil_level_percent2 = detect_oil_level_by_color(
            warped_cp, lower=None, upper=None, object_color=["yellow", "orange"], reduce_sum_thr=reduce_sum_thr, area_thresh=area_thresh, y_range=oil_range, color=color, thickness=thickness
        )
        if oil_level_percent < 0 or oil_level_percent > 100:
            oil_level_percent = None
        if oil_level_percent2 < 0 or oil_level_percent2 > 100:
            oil_level_percent2 = None

        if oil_level is not None and oil_level2 is not None:
            if abs(oil_level - oil_level2) < 10:
                return detected_img, oil_level2, oil_level_percent2
            else:
                print("两种算法检测结果偏差比较大，采用第二种方法！")
                return detected_img2, oil_level2, oil_level_percent2
        elif oil_level is None and oil_level2 is not None:
            return detected_img2, oil_level2, oil_level_percent2
        else:
            return detected_img, oil_level, oil_level_percent
    else:
        raise NotImplementedError



def main_test_20250410(data_path):
    dirname = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, '../{}_xfeat_matching_results'.format(dirname)))
    os.makedirs(save_path, exist_ok=True)

    # xfeat = XFeat()
    model_path = './weights/xfeat_matching.onnx'    # python ./export.py --dynamic --export_path ./xfeat_matching.onnx
    providers = [
        # The TensorrtExecutionProvider is the fastest.
        ('TensorrtExecutionProvider', { 
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_engine_cache',
            'trt_engine_cache_prefix': 'model',
            'trt_dump_subgraphs': False,
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': './trt_engine_cache',
            #'trt_builder_optimization_level': 3,
        }),

        # The CUDAExecutionProvider is slower than PyTorch, 
        # possibly due to performance issues with large matrix multiplication "cossim = torch.bmm(feats1, feats2.permute(0,2,1))"
        # Reducing the top_k value when exporting to ONNX can decrease the matrix size.
        ('CUDAExecutionProvider', { 
            'device_id': 0,
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
        }),
        ('CPUExecutionProvider',{ 
        })
    ]
    ort_session = ort.InferenceSession(model_path, providers=providers)


    # rect = np.array([[804, 651], [862, 644], [884, 917], [823, 924]], dtype='float32')
    rect = np.array([[779, 549], [871, 541], [903, 1013], [815, 1023]], dtype='float32')
    im1 = cv2.imread(r'G:\Gosion\data\008.OilLevel_Det\data\xfeat\ref.jpg')
    im1 = perspective_transform(im1, rect)

    file_list = sorted(os.listdir(data_path))
    for f in tqdm.tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = os.path.join(data_path, f)
        im2 = cv2.imread(f_abs_path)
        imgsz = im2.shape[:2]

        # mkpts_0, mkpts_1 = xfeat.match_xfeat_star(im1, img, top_k=8000)
        # canvas, warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, img)
        # warped_corners = np.asarray(warped_corners, dtype='float32').reshape(-1, 2)
        # f_dst_path = os.path.join(save_path, fname + '_xfeat_matching_result.jpg')
        # cv2.imwrite(f_dst_path, canvas)

        # input_array_1 = im1.transpose(2, 0, 1).astype(np.float32)
        # input_array_1 = np.expand_dims(input_array_1, axis=0)
        # input_array_2 = im2.transpose(2, 0, 1).astype(np.float32)
        # input_array_2 = np.expand_dims(input_array_2, axis=0)

        # inputs = {
        #     ort_session.get_inputs()[0].name: input_array_1,
        #     ort_session.get_inputs()[1].name: input_array_2
        # }

        # outputs = ort_session.run(None, inputs)

        # matches = outputs[0]
        # batch_indexes = outputs[1]
        # mkpts_0, mkpts_1 = matches[batch_indexes == 0][..., :2], matches[batch_indexes == 0][..., 2:]

    
        # canvas, warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)

        # flag = check_warped_corners(warped_corners, imgsz)
        # if not flag: continue

        # f_dst_path = os.path.join(save_path, fname + '_xfeat_matching_result.jpg')
        # cv2.imwrite(f_dst_path, canvas)

        # warped = perspective_transform(im2, np.array(warped_corners, dtype='float32').reshape(-1, 2))
        # warped_cp = warped.copy()
        # w_dst_path = os.path.join(save_path, fname + '_warped.jpg')
        # cv2.imwrite(w_dst_path, warped)

        # detected_img, oil_level = detect_oil_level_by_edge(warped, reduce_sum_thr=500, y_range=(115, 325), color=(255, 0, 255), thickness=4)
        # detected_oil_level_path = save_path + "/{}_oil_level.jpg".format(fname)
        # cv2.imwrite(detected_oil_level_path, detected_img)

        # detected_img2, oil_level2 = detect_oil_level_by_color(
        #     warped_cp, lower=None, upper=None, object_color=["yellow", "orange"], reduce_sum_thr=500, area_thresh=100, y_range=(115, 325), color=(255, 255, 0), thickness=4
        # )
        # detected_oil_level2_path = save_path + "/{}_oil_level2.jpg".format(fname)
        # cv2.imwrite(detected_oil_level2_path, detected_img2)

        
        # detected_img, oil_level, oil_level_percent = detect_oil_level_by_xfeat_onnx(ort_session, im1, im2, alg=1, oil_range=(115, 325), color=(255, 255, 0), thickness=4)
        # detected_img, oil_level, oil_level_percent = detect_oil_level_by_xfeat_onnx(ort_session, im1, im2, alg=1, oil_range=(0.2411, 0.6813), color=(255, 255, 0), thickness=4)
        detected_img, oil_level, oil_level_percent = detect_oil_level_by_xfeat_onnx(ort_session, im1, im2, alg=1, oil_range=(24.11, 68.13), reduce_sum_thr=500, area_thresh=100, color=(255, 0, 255), thickness=4)
        detected_oil_level_path = save_path + "/{}_oil_level_{}.jpg".format(fname, oil_level_percent)
        cv2.imwrite(detected_oil_level_path, detected_img)










if __name__ == '__main__':
    # model_path = './weights/xfeat_matching.onnx'    # python ./export.py --dynamic --export_path ./xfeat_matching.onnx

    # #Load some example images
    # im1 = cv2.imread('./assets/ref.png', cv2.IMREAD_COLOR)
    # im2 = cv2.imread('./assets/tgt.png', cv2.IMREAD_COLOR)

    # # tmp_ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # # # print the input,output names and shapes
    # # for i in range(len(tmp_ort_session.get_inputs())):
    # #     print(f"Input name: {tmp_ort_session.get_inputs()[i].name}, shape: {tmp_ort_session.get_inputs()[i].shape}")
    # # for i in range(len(tmp_ort_session.get_outputs())):
    # #     print(f"Output name: {tmp_ort_session.get_outputs()[i].name}, shape: {tmp_ort_session.get_outputs()[i].shape}")


    # providers = [
    #     # The TensorrtExecutionProvider is the fastest.
    #     ('TensorrtExecutionProvider', { 
    #         'device_id': 0,
    #         'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
    #         'trt_fp16_enable': True,
    #         'trt_engine_cache_enable': True,
    #         'trt_engine_cache_path': './trt_engine_cache',
    #         'trt_engine_cache_prefix': 'model',
    #         'trt_dump_subgraphs': False,
    #         'trt_timing_cache_enable': True,
    #         'trt_timing_cache_path': './trt_engine_cache',
    #         #'trt_builder_optimization_level': 3,
    #     }),

    #     # The CUDAExecutionProvider is slower than PyTorch, 
    #     # possibly due to performance issues with large matrix multiplication "cossim = torch.bmm(feats1, feats2.permute(0,2,1))"
    #     # Reducing the top_k value when exporting to ONNX can decrease the matrix size.
    #     ('CUDAExecutionProvider', { 
    #         'device_id': 0,
    #         'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
    #     }),
    #     ('CPUExecutionProvider',{ 
    #     })
    # ]
    # ort_session = ort.InferenceSession(model_path, providers=providers)

    # # im1 = cv2.resize(im1, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
    # # im2 = cv2.resize(im2, dsize=None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)

    # input_array_1 = im1.transpose(2, 0, 1).astype(np.float32)
    # input_array_1 = np.expand_dims(input_array_1, axis=0)
    # input_array_2 = im2.transpose(2, 0, 1).astype(np.float32)
    # input_array_2 = np.expand_dims(input_array_2, axis=0)

    # batch_size = 8

    # # Psuedo-batch the input images
    # input_array_1 = np.concatenate([input_array_1 for _ in range(batch_size)], axis=0)
    # input_array_2 = np.concatenate([input_array_2 for _ in range(batch_size)], axis=0)

    # inputs = {
    #     ort_session.get_inputs()[0].name: input_array_1,
    #     ort_session.get_inputs()[1].name: input_array_2
    # }

    # t1 = time.time()
    # outputs = ort_session.run(None, inputs)
    # t2 = time.time()
    # print(f"Inference time: {t2-t1}")

    # matches = outputs[0]
    # batch_indexes = outputs[1]
    # mkpts_0, mkpts_1 = matches[batch_indexes == 0][..., :2], matches[batch_indexes == 0][..., 2:]

    # canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
    # plt.figure(figsize=(12,12))
    # plt.imshow(canvas[..., ::-1])
    # plt.show()

    main_test_20250410(data_path=r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data")






