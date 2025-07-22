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


def draw_roi(img, roi_rect, color=(255, 0, 255), thickness=5):
    for i in range(len(roi_rect)):
        start_point = tuple(roi_rect[i-1].astype(int))
        end_point = tuple(roi_rect[i].astype(int))
        cv2.line(img, start_point, end_point, color, thickness)  # Using solid green for corners
    return img


def hstack_canvas(img1, img2):
    img1sz = img1.shape[:2]
    img2sz = img2.shape[:2]

    if img2sz[0] > img1sz[0]:
        img2 = cv2.resize(img2, (img2sz[1], img1sz[0]))
    else:
        padh = img1sz[0] - img2sz[0]
        z = np.zeros((padh, img2sz[1], 3), dtype=np.uint8)
        img2 = np.vstack((img2, z))
    
    hstack = np.hstack((img1, img2))

    return hstack


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


def detect(
        ort_session,
        im1,
        im2,
        roi_rect=None,
        reduce_sum_thr=500,
        area_thresh=100,
        color=(255, 255, 0),
        thickness=4
    ):
    """
    油位计检测：
    im1: 模板图
    im2: 测试图
    roi_rect: 目标区域, 4个点分别是[tl, tr, br, bl], e.g. np.array([[779, 549], [871, 541], [903, 1013], [815, 1023]], dtype='float32')

    alg: 算法类型
    alg 0: 通过边缘检测和连通域分析计算液面的位置
    alg 1: 通过颜色分析计算液面的位置
    alg 2: 综合0和1
    alg 3: 其他方法(备用, 例如深度学习算法)

    oil_range: 油位计量程, 油位计最高位和最低位位于 ROI小图中的y轴位置, 默认使用百分比形式, 也可以是小数形式(百分比除以100)和像素值y坐标形式
    reduce_sum_thr: alg 0使用到的一个参数, 默认为500
    area_thresh: 面积过滤阈值
    color: roi区域框的颜色
    thickness: roi区域框的线宽
    """
    im1_vis = im1.copy()

    if roi_rect is not None:
        assert type(roi_rect) == np.ndarray, "roi_rect should be np.ndarray!"
        im1 = perspective_transform(im1, roi_rect)
        im1_vis = draw_roi(im1_vis, roi_rect, color=(255, 0, 255), thickness=5)

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

    matched, warped_corners = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)

    im1_vis_sz = im1_vis.shape[:2]
    matched_sz = matched.shape[:2]
    if matched_sz[0] != im1_vis_sz[0]:
        matched = cv2.resize(matched, (matched_sz[1], im1_vis_sz[0]))
    canvas = np.hstack((im1_vis, matched))
    
    flag = check_warped_corners(warped_corners, im2.shape[:2])
    if not flag:
        return canvas, im2, None, None

    warped = perspective_transform(im2, np.array(warped_corners, dtype='float32').reshape(-1, 2))
    warpedsz = warped.shape[:2]
    warped_cp = warped.copy()
    

def main():
    dirname = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, '../{}_xfeat_matching_results'.format(dirname)))
    os.makedirs(save_path, exist_ok=True)

    # xfeat = XFeat()
    model_path = r'D:\GraceKafuu\Python\github\gkfutils\gkfutils\cv\xfeat\weights\xfeat_matching.onnx'    # python ./export.py --dynamic --export_path ./xfeat_matching.onnx
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
    im1 = cv2.imread(r'G:\Gosion\data\008.Oil_Level_Det\data\xfeat\ref.jpg')

    file_list = sorted(os.listdir(data_path))
    for f in tqdm.tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = os.path.join(data_path, f)
        im2 = cv2.imread(f_abs_path)
        imgsz = im2.shape[:2]

        matched, res, l, lp = detect_oil_level_by_xfeat_onnx_v2(ort_session, im1, im2, roi_rect=rect, alg=0, oil_range=(24.11, 68.13), reduce_sum_thr=500, area_thresh=100, color=(255, 0, 255), thickness=4)
        matched_path = save_path + "/{}_matched_oil_level_{}.jpg".format(fname, lp)
        cv2.imwrite(matched_path, matched)


if __name__ == '__main__':
    main()