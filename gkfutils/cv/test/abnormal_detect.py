import numpy as np
import matplotlib.pyplot as plt

def z_score_detection(data, threshold=3):
    """
    Z-Score异常检测
    适用于近似正态分布的数据
    """
    mean = np.mean(data)
    std = np.std(data)
    # z_scores = np.abs((data - mean) / std)
    # anomalies = np.where(z_scores > 3 * std)[0]
    z_scores = (data - mean) / std
    # anomalies = np.where((z_scores < (mean - std)) | (z_scores > (mean + std)))[0]
    anomalies = np.where((z_scores < (mean - 2 * std)) | (z_scores > (mean + 2 * std)))[0]
    # anomalies = np.where((z_scores < (mean - 3 * std)) | (z_scores > (mean + 3 * std)))[0]
    return anomalies, z_scores


def iqr_detection(data, multiplier=1.5):
    """
    IQR方法 - 对非正态分布数据更稳健
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]
    return anomalies, (lower_bound, upper_bound)


# 示例
np.random.seed(42)
# normal_data = np.random.normal(0, 1, 1000)
# data_with_outliers = np.concatenate([normal_data, [5, -5, 4.8, -4.9]])

# data_with_outliers = np.array([5, 5, 5.9, 4.9, 5.3, 5.2, 1, 0, 5.2, 5.5, 5, 6, 5, 5.5, 10, 12, 10, 11])
# data_with_outliers = np.array([5, 5, 5, 5, 0, 0, 5, 5, 5, 5])
data_with_outliers = np.array([5, 5, 5, 5, 0, 0, 0, 5, 5, 5, 5])
# data_with_outliers = np.array([5, 5, 5.9, 4.9, 5.3, 5.2, 1, 1, 0, 5.2, 5.5, 5, 6, 5, 5.5, 5.8, 5.7, 5.6, 10 ,12, 23, 12, 14, 15, 5., 5, 5, 5, 5., 5, 5, 5, 5., 5, 5, 5, 5., 5, 5, 5, 5., 5, 5, 5, 5., 5, 5, 5])


anomalies, z_scores = z_score_detection(data_with_outliers, threshold=1.0)
print(f"Z-Score检测到的异常索引: {anomalies}")


anomalies_iqr, bounds = iqr_detection(data_with_outliers)
print(f"IQR检测到的异常索引: {anomalies_iqr}")