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


def main():
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
    # rect = np.array([[779, 549], [871, 541], [903, 1013], [815, 1023]], dtype='float32')
    # im1 = cv2.imread(r'G:\Gosion\data\008.OilLevel_Det\data\xfeat\ref.jpg')
    # im1 = perspective_transform(im1, rect)

    im1 = cv2.imread(r"D:\GraceKafuu\Python\github\gkfutils\gkfutils\cv\xfeat\data\ref.png")
    im2 = cv2.imread(r"D:\GraceKafuu\Python\github\gkfutils\gkfutils\cv\xfeat\data\tgt.png")
    
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
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()