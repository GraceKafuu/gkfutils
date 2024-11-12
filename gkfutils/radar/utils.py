import os
import cv2
import numpy as np


# PCL
def mat2pcl(img, K):
    import open3d as o3d

    f_x = K[0, 0]
    f_y = K[1, 1]
    s = K[0, 1]
    p_x = K[0, 2]
    p_y = K[1, 2]

    y_values, x_values = np.where(img > 300)
    z_values = img[y_values, x_values]

    Y_values = (y_values - p_y) * z_values / f_y
    X_values = (x_values - p_x - (y_values - p_y) * s / f_y) * z_values / f_x

    points_3d = np.array([X_values, Y_values, z_values]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    return pcd


def main_pcl():
    import open3d as o3d

    fs = cv2.FileStorage(r"data/matrix.yml", cv2.FileStorage_READ)
    K = fs.getNode("M").mat()
    fs.release()

    image_path = r"data/2111043208/1-1067.data"
    h_image = cv2.imread(image_path, 2)

    # h_image = (h_image/65535)*900+200

    pcd = mat2pcl(h_image, K)
    o3d.io.write_point_cloud(image_path.replace(".data", ".ply"), pcd)

    return 0


if __name__ == '__main__':
    print("OK!")