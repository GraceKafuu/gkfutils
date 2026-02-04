# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision import transforms
# import os
# from tqdm import tqdm
# import cv2
# from sklearn.metrics import mean_absolute_error

# # ==================== 1. 数据生成器 ====================
# class HorizontalLineRegressionDataset(Dataset):
#     """生成带随机水平直线的图像数据集，标签为y坐标值"""
#     def __init__(self, num_samples=10000, img_size=(256, 256), 
#                  line_width_range=(1, 5), line_intensity_range=(0.3, 1.0),
#                  noise_level=0.1, train=True, transform=None):
#         self.num_samples = num_samples
#         self.img_size = img_size
#         self.line_width_range = line_width_range
#         self.line_intensity_range = line_intensity_range
#         self.noise_level = noise_level
#         self.train = train
#         self.transform = transform
        
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # 创建随机背景
#         if self.train:
#             # 训练时使用更多样的背景
#             background_type = np.random.choice(['uniform', 'gradient', 'texture', 'noise'])
            
#             if background_type == 'uniform':
#                 img = np.ones(self.img_size) * np.random.uniform(0.2, 0.8)
#             elif background_type == 'gradient':
#                 # 创建垂直渐变
#                 gradient = np.linspace(0.3, 0.7, self.img_size[0]).reshape(-1, 1)
#                 img = np.tile(gradient, (1, self.img_size[1]))
#             elif background_type == 'texture':
#                 # 随机纹理
#                 img = np.random.randn(*self.img_size) * 0.1 + 0.5
#                 # 添加一些低频变化
#                 xx, yy = np.meshgrid(np.linspace(0, 1, self.img_size[1]), 
#                                     np.linspace(0, 1, self.img_size[0]))
#                 img += 0.2 * np.sin(2 * np.pi * (np.random.rand() * xx + np.random.rand() * yy))
#             else:  # noise
#                 img = np.random.rand(*self.img_size) * 0.5 + 0.25
#         else:
#             # 验证/测试时使用更简单的背景
#             img = np.ones(self.img_size) * 0.5
        
#         # 归一化背景到[0, 1]
#         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
#         # 随机生成直线参数
#         line_y = np.random.randint(20, self.img_size[0] - 20)  # 确保直线不贴边
#         line_width = np.random.randint(*self.line_width_range)
#         line_intensity = np.random.uniform(*self.line_intensity_range)
        
#         # 确保直线比背景更亮或更暗
#         if np.random.rand() > 0.5:  # 50%概率直线比背景亮
#             line_intensity = min(1.0, img.mean() + line_intensity)
#         else:  # 50%概率直线比背景暗
#             line_intensity = max(0.0, img.mean() - line_intensity)
        
#         # 添加水平直线
#         half_width = line_width // 2
#         start_row = max(0, line_y - half_width)
#         end_row = min(self.img_size[0], line_y + line_width - half_width)
        
#         # 创建直线（可添加一些不连续性模拟真实情况）
#         if self.train and np.random.rand() < 0.3:  # 30%概率直线有间断
#             gap_start = np.random.randint(0, self.img_size[1] - 50)
#             gap_length = np.random.randint(10, 50)
            
#             # 直线前段
#             img[start_row:end_row, :gap_start] = line_intensity
            
#             # 直线后段
#             gap_end = gap_start + gap_length
#             img[start_row:end_row, gap_end:] = line_intensity
            
#             # 间断处保持背景
#             if self.train and np.random.rand() < 0.5:  # 间断处可能有一些残留
#                 img[start_row:end_row, gap_start:gap_end] = line_intensity * 0.3
#         else:
#             # 连续直线
#             img[start_row:end_row, :] = line_intensity
        
#         # 添加高斯噪声
#         noise = np.random.randn(*self.img_size) * self.noise_level
#         img = img + noise
        
#         # 添加其他干扰（训练时）
#         if self.train:
#             # 随机添加少量垂直或斜线
#             if np.random.rand() < 0.1:
#                 x_pos = np.random.randint(0, self.img_size[1])
#                 img[:, max(0, x_pos-1):min(self.img_size[1], x_pos+2)] += np.random.uniform(0.2, 0.4)
            
#             # 添加随机点状噪声
#             num_points = np.random.randint(5, 20)
#             for _ in range(num_points):
#                 px = np.random.randint(0, self.img_size[1])
#                 py = np.random.randint(0, self.img_size[0])
#                 radius = np.random.randint(1, 3)
#                 img[max(0, py-radius):min(self.img_size[0], py+radius+1), 
#                     max(0, px-radius):min(self.img_size[1], px+radius+1)] += np.random.uniform(0.3, 0.6)
        
#         # 归一化到[0, 1]
#         img = np.clip(img, 0, 1)
        
#         # 标准化（零均值，单位方差）
#         img = (img - img.mean()) / (img.std() + 1e-8)
        
#         # 转换为张量
#         img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        
#         # 归一化标签到[0, 1]范围，便于回归
#         label_normalized = line_y / self.img_size[0]
#         label_tensor = torch.FloatTensor([label_normalized])
        
#         return img_tensor, label_tensor, line_y

# # ==================== 2. 回归网络结构 ====================
# class HorizontalLineRegressor(nn.Module):
#     """回归网络：预测直线的y坐标"""
#     def __init__(self, input_size=(1, 256, 256)):
#         super().__init__()
#         self.input_size = input_size
        
#         # 特征提取部分 - 使用多个卷积层捕获空间特征
#         self.conv_layers = nn.Sequential(
#             # Block 1
#             nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 128x128
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            
#             # Block 2
#             nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 64x64
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
            
#             # Block 3
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 32x32
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
#             # Block 4
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 16x16
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            
#             # Block 5
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 8x8
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((4, 4)),  # 固定输出大小
#         )
        
#         # 全连接回归部分
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 4 * 4, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
            
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(128, 1),
#             nn.Sigmoid()  # 输出在[0, 1]范围内
#         )
        
#     def forward(self, x):
#         features = self.conv_layers(x)
#         output = self.fc_layers(features)
#         return output

# class EnhancedLineRegressor(nn.Module):
#     """增强版回归网络：使用空间注意力和多尺度特征"""
#     def __init__(self, input_size=(1, 256, 256)):
#         super().__init__()
        
#         # 多尺度特征提取
#         # 路径1: 高分辨率路径（用于精确定位）
#         self.high_res_path = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 128x128
#         )
        
#         # 路径2: 中分辨率路径
#         self.mid_res_path = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # 128x128
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 64x64
#         )
        
#         # 路径3: 低分辨率路径（大感受野）
#         self.low_res_path = nn.Sequential(
#             nn.Conv2d(1, 128, kernel_size=7, stride=4, padding=3),  # 64x64
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 32x32
#         )
        
#         # 空间注意力模块
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(32 + 64 + 128, 64, kernel_size=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, kernel_size=1),
#             nn.Sigmoid()  # 注意力权重在[0, 1]范围
#         )
        
#         # 特征融合后的卷积
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(32 + 64 + 128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 16x16
            
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((4, 4)),
#         )
        
#         # 回归头
#         self.regression_head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 4 * 4, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
            
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
            
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
            
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # 多尺度特征提取
#         high_res = self.high_res_path(x)  # 128x128
#         mid_res = self.mid_res_path(x)    # 64x64
#         low_res = self.low_res_path(x)    # 32x32
        
#         # 上采样到统一尺寸(32x32)
#         high_res_up = F.interpolate(high_res, size=(32, 32), mode='bilinear', align_corners=False)
#         mid_res_up = F.interpolate(mid_res, size=(32, 32), mode='bilinear', align_corners=False)
        
#         # 特征拼接
#         combined = torch.cat([high_res_up, mid_res_up, low_res], dim=1)
        
#         # 应用空间注意力
#         attention_weights = self.spatial_attention(combined)
#         attended_features = combined * attention_weights
        
#         # 特征融合
#         fused_features = self.fusion_conv(attended_features)
        
#         # 回归
#         output = self.regression_head(fused_features)
        
#         return output, attention_weights

# # ==================== 3. 训练函数 ====================
# class LineRegressionTrainer:
#     def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
#                  use_enhanced=True):
#         self.model = model.to(device)
#         self.device = device
#         self.use_enhanced = use_enhanced
        
#         # 使用Smooth L1 Loss（Huber Loss），对异常值不敏感
#         self.criterion = nn.SmoothL1Loss(beta=0.1)
        
#         # 也可以使用MAE Loss
#         # self.criterion = nn.L1Loss()
        
#         self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
#         # 使用余弦退火学习率调度
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             self.optimizer, T_max=50, eta_min=1e-5
#         )
        
#         # 学习率预热
#         self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#             self.optimizer, start_factor=0.1, total_iters=5
#         )
        
#         # 组合调度器
#         self.combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
#             self.optimizer,
#             schedulers=[self.warmup_scheduler, self.scheduler],
#             milestones=[5]
#         )
        
#         self.best_val_mae = float('inf')
        
#     def train_epoch(self, dataloader, epoch):
#         self.model.train()
#         total_loss = 0
#         all_predictions = []
#         all_labels = []
        
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
#         for batch_idx, (images, labels_norm, labels_orig) in enumerate(pbar):
#             images = images.to(self.device)
#             labels = labels_norm.to(self.device)
            
#             # 前向传播
#             if self.use_enhanced:
#                 predictions, _ = self.model(images)
#             else:
#                 predictions = self.model(images)
            
#             # 计算损失
#             loss = self.criterion(predictions, labels)
            
#             # 反向传播
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
#             self.optimizer.step()
            
#             total_loss += loss.item()
            
#             # 收集预测结果用于计算指标
#             all_predictions.extend(predictions.detach().cpu().numpy().flatten())
#             all_labels.extend(labels_norm.cpu().numpy().flatten())
            
#             # 更新进度条
#             pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
#         # 计算MAE
#         mae = mean_absolute_error(all_labels, all_predictions)
        
#         return total_loss / len(dataloader), mae
    
#     def validate(self, dataloader):
#         self.model.eval()
#         total_loss = 0
#         all_predictions = []
#         all_labels = []
#         all_labels_orig = []
        
#         with torch.no_grad():
#             pbar = tqdm(dataloader, desc="Validation")
#             for images, labels_norm, labels_orig in pbar:
#                 images = images.to(self.device)
#                 labels = labels_norm.to(self.device)
                
#                 if self.use_enhanced:
#                     predictions, _ = self.model(images)
#                 else:
#                     predictions = self.model(images)
                
#                 loss = self.criterion(predictions, labels)
#                 total_loss += loss.item()
                
#                 # 收集结果
#                 all_predictions.extend(predictions.cpu().numpy().flatten())
#                 all_labels.extend(labels_norm.cpu().numpy().flatten())
#                 all_labels_orig.extend(labels_orig.cpu().numpy().flatten())
                
#                 pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
#         # 转换为numpy数组
#         all_predictions = np.array(all_predictions)
#         all_labels = np.array(all_labels)
#         all_labels_orig = np.array(all_labels_orig)
        
#         # 计算归一化空间的MAE
#         mae_norm = mean_absolute_error(all_labels, all_predictions)
        
#         # 转换回原始像素空间的预测
#         img_height = 256  # 假设图像高度为256
#         predictions_pixel = all_predictions * img_height
#         mae_pixel = mean_absolute_error(all_labels_orig, predictions_pixel)
        
#         return total_loss / len(dataloader), mae_norm, mae_pixel
    
#     def train(self, train_loader, val_loader, epochs=50):
#         train_losses = []
#         val_losses = []
#         train_maes = []
#         val_maes_pixel = []
        
#         print(f"Starting training for {epochs} epochs...")
#         print(f"Using device: {self.device}")
#         print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
#         for epoch in range(epochs):
#             print(f"\n{'='*60}")
#             print(f"Epoch {epoch+1}/{epochs}")
#             print(f"{'='*60}")
            
#             # 训练
#             train_loss, train_mae = self.train_epoch(train_loader, epoch)
#             train_losses.append(train_loss)
#             train_maes.append(train_mae)
            
#             # 验证
#             val_loss, val_mae_norm, val_mae_pixel = self.validate(val_loader)
#             val_losses.append(val_loss)
#             val_maes_pixel.append(val_mae_pixel)
            
#             # 更新学习率
#             self.combined_scheduler.step()
            
#             # 打印结果
#             current_lr = self.optimizer.param_groups[0]['lr']
#             print(f"\nResults:")
#             print(f"  Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}")
#             print(f"  Val Loss: {val_loss:.6f}, Val MAE (pixel): {val_mae_pixel:.2f}")
#             print(f"  Learning Rate: {current_lr:.6f}")
            
#             # 保存最佳模型
#             if val_mae_pixel < self.best_val_mae:
#                 self.best_val_mae = val_mae_pixel
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                     'scheduler_state_dict': self.combined_scheduler.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                     'val_mae_pixel': val_mae_pixel,
#                     'best_val_mae': self.best_val_mae,
#                 }, 'best_line_regressor.pth')
#                 print(f"  ✓ Saved best model with pixel MAE: {val_mae_pixel:.2f}")
            
#             # 每10个epoch保存一次检查点
#             if (epoch + 1) % 10 == 0:
#                 checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                     'train_loss': train_loss,
#                     'val_loss': val_loss,
#                 }, checkpoint_path)
#                 print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
#         # 训练完成
#         print(f"\n{'='*60}")
#         print("Training completed!")
#         print(f"Best pixel MAE: {self.best_val_mae:.2f}")
        
#         # 绘制训练曲线
#         self.plot_training_curves(train_losses, val_losses, train_maes, val_maes_pixel)
        
#         return train_losses, val_losses, train_maes, val_maes_pixel
    
#     def plot_training_curves(self, train_losses, val_losses, train_maes, val_maes_pixel):
#         fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
#         # 损失曲线
#         axes[0].plot(train_losses, label='Training Loss', linewidth=2)
#         axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
#         axes[0].set_xlabel('Epoch')
#         axes[0].set_ylabel('Loss')
#         axes[0].set_title('Training and Validation Loss')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
        
#         # MAE曲线
#         axes[1].plot(train_maes, label='Training MAE (normalized)', linewidth=2)
#         axes[1].plot(val_maes_pixel, label='Validation MAE (pixels)', linewidth=2)
#         axes[1].set_xlabel('Epoch')
#         axes[1].set_ylabel('MAE')
#         axes[1].set_title('Training and Validation MAE')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
#         plt.show()

# # ==================== 4. 检测器类 ====================
# class LineRegressionDetector:
#     """直线回归检测器"""
#     def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu',
#                  use_enhanced=True, img_size=(256, 256)):
#         self.device = device
#         self.img_size = img_size
        
#         # 选择模型
#         if use_enhanced:
#             self.model = EnhancedLineRegressor(input_size=(1, *img_size))
#         else:
#             self.model = HorizontalLineRegressor(input_size=(1, *img_size))
        
#         # 加载模型权重
#         if model_path and os.path.exists(model_path):
#             checkpoint = torch.load(model_path, map_location=device)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             print(f"Loaded model from {model_path}")
#             if 'best_val_mae' in checkpoint:
#                 print(f"Best validation MAE: {checkpoint['best_val_mae']:.2f} pixels")
#         else:
#             print("Warning: No model weights loaded. Using untrained model.")
        
#         self.model.to(device)
#         self.model.eval()
    
#     def preprocess_image(self, image):
#         """预处理输入图像"""
#         if isinstance(image, np.ndarray):
#             # 已经是numpy数组
#             if len(image.shape) == 3 and image.shape[2] == 3:
#                 # 彩色转灰度
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#             elif len(image.shape) == 2:
#                 # 已经是灰度图
#                 pass
#             else:
#                 raise ValueError(f"Unsupported image shape: {image.shape}")
#         elif isinstance(image, Image.Image):
#             # PIL图像
#             image = np.array(image.convert('L'))
#         else:
#             raise TypeError("Input must be numpy array or PIL Image")
        
#         # 调整大小
#         if image.shape != self.img_size:
#             image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        
#         # 归一化
#         image = image.astype(np.float32) / 255.0
        
#         # 标准化
#         image = (image - image.mean()) / (image.std() + 1e-8)
        
#         # 转换为张量
#         image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
#         return image_tensor, image
    
#     def detect_line(self, image, return_confidence=False):
#         """
#         检测图像中的水平直线
        
#         Args:
#             image: 输入图像（numpy数组或PIL Image）
#             return_confidence: 是否返回置信度
        
#         Returns:
#             line_y: 直线y坐标（原始像素空间）
#             confidence: 置信度（可选）
#         """
#         # 预处理
#         image_tensor, processed_image = self.preprocess_image(image)
#         image_tensor = image_tensor.to(self.device)
        
#         with torch.no_grad():
#             # 前向传播
#             if isinstance(self.model, EnhancedLineRegressor):
#                 prediction, attention_weights = self.model(image_tensor)
#             else:
#                 prediction = self.model(image_tensor)
#                 attention_weights = None
            
#             # 获取预测值
#             y_norm = prediction.item()
            
#             # 转换回原始像素空间
#             line_y = int(y_norm * self.img_size[0])
            
#             # 计算置信度（基于预测值的合理性）
#             if return_confidence:
#                 # 简单的置信度：距离边界越远，置信度越高
#                 margin = min(line_y, self.img_size[0] - line_y)
#                 confidence = min(1.0, margin / (self.img_size[0] / 4))
                
#                 return line_y, confidence, attention_weights
#             else:
#                 return line_y, attention_weights
    
#     def visualize_detection(self, image, save_path=None):
#         """可视化检测结果"""
#         # 检测直线
#         line_y, confidence, attention_weights = self.detect_line(image, return_confidence=True)
        
#         # 获取原始图像
#         if isinstance(image, np.ndarray):
#             orig_image = image.copy()
#             if len(orig_image.shape) == 2:
#                 orig_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2RGB)
#         else:
#             orig_image = np.array(image.convert('RGB'))
        
#         # 调整大小用于显示
#         display_image = cv2.resize(orig_image, (self.img_size[1], self.img_size[0]))
        
#         # 创建可视化
#         fig, axes = plt.subplots(1, 3 if attention_weights is not None else 2, 
#                                 figsize=(15, 5))
        
#         # 原始图像
#         axes[0].imshow(display_image)
#         axes[0].axhline(y=line_y, color='red', linewidth=3, linestyle='--', 
#                        label=f'Detected: y={line_y}')
#         axes[0].set_title(f'Original Image\nConfidence: {confidence:.2f}')
#         axes[0].legend()
#         axes[0].axis('off')
        
#         # 预处理后的图像
#         _, processed_image = self.preprocess_image(image)
#         axes[1].imshow(processed_image, cmap='gray')
#         axes[1].axhline(y=line_y, color='yellow', linewidth=2)
#         axes[1].set_title('Preprocessed Image')
#         axes[1].axis('off')
        
#         # 注意力图（如果有）
#         if attention_weights is not None:
#             att_map = attention_weights.squeeze().cpu().numpy()
#             att_map = cv2.resize(att_map, (self.img_size[1], self.img_size[0]))
            
#             axes[2].imshow(att_map, cmap='hot')
#             axes[2].set_title('Attention Map')
#             axes[2].axis('off')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.show()
        
#         return line_y, confidence

# # ==================== 5. 测试函数 ====================
# def test_model_performance(model_path, num_tests=100):
#     """测试模型性能"""
#     print(f"\n{'='*60}")
#     print("Testing Model Performance")
#     print(f"{'='*60}")
    
#     # 创建测试数据集
#     test_dataset = HorizontalLineRegressionDataset(num_samples=num_tests, train=False)
    
#     # 创建检测器
#     detector = LineRegressionDetector(model_path)
    
#     errors = []
    
#     for i in range(min(10, num_tests)):  # 只测试前10个用于可视化
#         img, label_norm, true_y = test_dataset[i]
        
#         # 转换为PIL图像用于检测器
#         img_np = img.squeeze().numpy()
#         img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
#         # 检测直线
#         pred_y, confidence = detector.detect_line(img_pil, return_confidence=False)
        
#         # 计算误差
#         error = abs(pred_y - true_y)
#         errors.append(error)
        
#         # 可视化前几个结果
#         if i < 3:
#             print(f"Test {i+1}: True y={true_y}, Predicted y={pred_y}, Error={error} pixels")
#             detector.visualize_detection(img_pil, save_path=f'test_result_{i}.png')
    
#     # 计算总体统计
#     errors = np.array(errors[:num_tests])
#     print(f"\nPerformance Statistics ({num_tests} test samples):")
#     print(f"  Mean Absolute Error: {errors.mean():.2f} pixels")
#     print(f"  Standard Deviation: {errors.std():.2f} pixels")
#     print(f"  Max Error: {errors.max():.2f} pixels")
#     print(f"  90th Percentile: {np.percentile(errors, 90):.2f} pixels")
    
#     # 绘制误差分布
#     plt.figure(figsize=(10, 6))
#     plt.hist(errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
#     plt.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
#                 label=f'Mean: {errors.mean():.2f}')
#     plt.xlabel('Absolute Error (pixels)')
#     plt.ylabel('Frequency')
#     plt.title('Error Distribution')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
#     plt.show()

# # ==================== 6. 主训练脚本 ====================
# def main():
#     # 设置随机种子
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     # 检查设备
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     # 创建数据集
#     print("\nCreating datasets...")
#     train_dataset = HorizontalLineRegressionDataset(
#         num_samples=10000, 
#         img_size=(256, 256),
#         line_width_range=(1, 5),
#         noise_level=0.1,
#         train=True
#     )
    
#     val_dataset = HorizontalLineRegressionDataset(
#         num_samples=2000,
#         img_size=(256, 256),
#         line_width_range=(1, 5),
#         noise_level=0.05,
#         train=False
#     )
    
#     # 创建数据加载器
#     batch_size = 32 if device == 'cuda' else 16
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=4,
#         pin_memory=True if device == 'cuda' else False
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=4,
#         pin_memory=True if device == 'cuda' else False
#     )
    
#     # 创建模型（选择增强版或基础版）
#     use_enhanced = True  # 设为False使用基础版
#     if use_enhanced:
#         print("Using EnhancedLineRegressor model")
#         model = EnhancedLineRegressor(input_size=(1, 256, 256))
#     else:
#         print("Using HorizontalLineRegressor model")
#         model = HorizontalLineRegressor(input_size=(1, 256, 256))
    
#     # 创建训练器
#     trainer = LineRegressionTrainer(model, device, use_enhanced=use_enhanced)
    
#     # 开始训练
#     epochs = 50
#     print(f"\nStarting training for {epochs} epochs...")
#     train_losses, val_losses, train_maes, val_maes_pixel = trainer.train(
#         train_loader, val_loader, epochs=epochs
#     )
    
#     print("\nTraining completed!")
    
#     # 测试模型
#     test_model_performance('best_line_regressor.pth', num_tests=100)

# # ==================== 7. 使用真实图像 ====================
# def detect_line_in_real_image(image_path, model_path='best_line_regressor.pth'):
#     """在真实图像中检测水平直线"""
#     # 加载图像
#     if not os.path.exists(image_path):
#         print(f"Error: Image not found at {image_path}")
#         return None
    
#     # 读取图像
#     image = Image.open(image_path)
    
#     # 创建检测器
#     detector = LineRegressionDetector(model_path)
    
#     # 检测直线
#     line_y, confidence = detector.visualize_detection(image, save_path='real_image_detection.png')
    
#     print(f"\nDetection Result:")
#     print(f"  Line y-coordinate: {line_y} pixels")
#     print(f"  Confidence: {confidence:.2f}")
    
#     return line_y

# if __name__ == "__main__":
#     # 训练模型
#     main()
    
#     # 使用示例（训练后）
#     # 检测真实图像中的直线
#     # detect_line_in_real_image('your_image.jpg')




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 直线检测数据集 ====================
class HorizontalLineDataset(Dataset):
    """生成带随机水平直线的图像数据集"""
    def __init__(self, num_samples=10000, img_size=(256, 256), 
                 line_width_range=(1, 5), train=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.line_width_range = line_width_range
        self.train = train
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 创建背景 - 多种背景类型增加鲁棒性
        bg_type = np.random.choice(['uniform', 'gradient', 'noise', 'texture'])
        
        if bg_type == 'uniform':
            bg_value = np.random.uniform(0.3, 0.7)
            img = np.ones(self.img_size) * bg_value
        elif bg_type == 'gradient':
            # 创建渐变背景
            gradient = np.linspace(0.2, 0.8, self.img_size[0]).reshape(-1, 1)
            img = np.tile(gradient, (1, self.img_size[1]))
            # 添加一些水平变化
            horiz = np.linspace(0.1, 0.3, self.img_size[1]).reshape(1, -1)
            img += np.tile(horiz, (self.img_size[0], 1))
        elif bg_type == 'noise':
            img = np.random.randn(*self.img_size) * 0.15 + 0.5
        else:  # texture
            img = np.random.rand(*self.img_size) * 0.4 + 0.3
            # 添加低频纹理
            x = np.linspace(0, 2*np.pi, self.img_size[1])
            y = np.linspace(0, 2*np.pi, self.img_size[0])
            xx, yy = np.meshgrid(x, y)
            img += 0.15 * np.sin(xx) * np.cos(yy)
        
        # 随机生成直线参数
        line_y = np.random.randint(20, self.img_size[0] - 20)
        line_width = np.random.randint(*self.line_width_range)
        
        # 直线对比度 - 随机选择亮线或暗线
        if np.random.rand() > 0.5:
            line_intensity = min(1.0, img.mean() + np.random.uniform(0.3, 0.8))
        else:
            line_intensity = max(0.0, img.mean() - np.random.uniform(0.3, 0.8))
        
        # 创建直线 - 可能有间断
        half_width = line_width // 2
        start_row = max(0, line_y - half_width)
        end_row = min(self.img_size[0], line_y + line_width - half_width)
        
        if self.train and np.random.rand() < 0.2:
            # 有间断的直线
            gap_start = np.random.randint(50, self.img_size[1] - 100)
            gap_width = np.random.randint(20, 60)
            img[start_row:end_row, :gap_start] = line_intensity
            img[start_row:end_row, gap_start+gap_width:] = line_intensity
        else:
            # 连续直线
            img[start_row:end_row, :] = line_intensity
        
        # 添加噪声
        noise = np.random.randn(*self.img_size) * 0.1
        img += noise
        
        # 添加干扰元素（训练时）
        if self.train:
            # 随机点状干扰
            num_dots = np.random.randint(5, 15)
            for _ in range(num_dots):
                px = np.random.randint(0, self.img_size[1])
                py = np.random.randint(0, self.img_size[0])
                radius = np.random.randint(1, 3)
                intensity = np.random.uniform(0.2, 0.6)
                py_start = max(0, py - radius)
                py_end = min(self.img_size[0], py + radius + 1)
                px_start = max(0, px - radius)
                px_end = min(self.img_size[1], px + radius + 1)
                img[py_start:py_end, px_start:px_end] += intensity
            
            # 偶尔添加垂直线干扰
            if np.random.rand() < 0.1:
                x_pos = np.random.randint(20, self.img_size[1] - 20)
                width = np.random.randint(1, 3)
                intensity = np.random.uniform(0.3, 0.6)
                img[:, x_pos:x_pos+width] += intensity
        
        # 裁剪并归一化
        img = np.clip(img, 0, 1)
        
        # 标准化
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # 转换为张量
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        
        # 归一化标签到[0, 1]
        label_normalized = line_y / self.img_size[0]
        label_tensor = torch.FloatTensor([label_normalized])
        
        return img_tensor, label_tensor, line_y

# ==================== 2. 直线检测网络 ====================
class LineDetectionCNN(nn.Module):
    """直线检测CNN - 输出单个y坐标值"""
    def __init__(self):
        super().__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # 固定尺寸
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出在[0, 1]范围内
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output

# ==================== 3. 直线检测训练器 ====================
class LineDetectionTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 使用Huber Loss（Smooth L1 Loss），对异常值更鲁棒
        self.criterion = nn.SmoothL1Loss(beta=0.1)
        
        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50,
            eta_min=1e-6
        )
        
        # 预热调度器
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=5
        )
        
        # 组合调度器
        self.combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.scheduler],
            milestones=[5]
        )
        
        self.best_val_mae = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        predictions = []
        ground_truth = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
        for batch_idx, (images, labels_norm, labels_orig) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels_norm.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测结果
            predictions.extend(outputs.detach().cpu().numpy().flatten())
            ground_truth.extend(labels_norm.numpy().flatten())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算MAE
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        mae = mean_absolute_error(ground_truth, predictions)
        
        return total_loss / len(dataloader), mae
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predictions = []
        ground_truth = []
        ground_truth_pixel = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for images, labels_norm, labels_orig in pbar:
                images = images.to(self.device)
                labels = labels_norm.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 收集结果
                predictions.extend(outputs.cpu().numpy().flatten())
                ground_truth.extend(labels_norm.numpy().flatten())
                ground_truth_pixel.extend(labels_orig.numpy().flatten())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        ground_truth_pixel = np.array(ground_truth_pixel)
        
        # 归一化空间的MAE
        mae_norm = mean_absolute_error(ground_truth, predictions)
        
        # 像素空间的MAE
        predictions_pixel = predictions * 256  # 假设图像高度为256
        mae_pixel = mean_absolute_error(ground_truth_pixel, predictions_pixel)
        
        # RMSE
        rmse_pixel = np.sqrt(mean_squared_error(ground_truth_pixel, predictions_pixel))
        
        return total_loss / len(dataloader), mae_norm, mae_pixel, rmse_pixel
    
    def train(self, train_loader, val_loader, epochs=50, save_dir='./line_detection'):
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss, train_mae = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_maes.append(train_mae)
            
            # 验证
            val_loss, val_mae_norm, val_mae_pixel, val_rmse_pixel = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae_pixel)
            
            # 更新学习率
            self.combined_scheduler.step()
            
            # 打印结果
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}")
            print(f"  Val Loss: {val_loss:.6f}, Val MAE: {val_mae_pixel:.2f}px, Val RMSE: {val_rmse_pixel:.2f}px")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_mae_pixel < self.best_val_mae:
                self.best_val_mae = val_mae_pixel
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae_pixel,
                    'best_val_mae': self.best_val_mae,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ Saved best model (MAE: {val_mae_pixel:.2f}px)")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        # 训练完成
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best validation MAE: {self.best_val_mae:.2f} pixels")
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        return self.train_losses, self.val_losses, self.train_maes, self.val_maes
    
    def plot_training_curves(self, save_dir):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Training Loss', linewidth=2, marker='o', markersize=4)
        axes[0].plot(self.val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE曲线
        axes[1].plot(self.train_maes, label='Training MAE', linewidth=2, marker='o', markersize=4)
        axes[1].plot(self.val_maes, label='Validation MAE (pixels)', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.show()

# ==================== 4. 直线检测器 ====================
class LineDetector:
    """直线检测器"""
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = LineDetectionCNN()
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            if 'best_val_mae' in checkpoint:
                print(f"Best validation MAE: {checkpoint['best_val_mae']:.2f} pixels")
        else:
            print("Warning: Using untrained model")
        
        self.model.to(device)
        self.model.eval()
    
    def preprocess(self, image):
        """预处理输入图像"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        else:
            raise TypeError("Input must be numpy array or PIL Image")
        
        # 调整大小到256x256
        image = cv2.resize(image, (256, 256))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 标准化
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # 转换为张量
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return image_tensor, image
    
    def detect(self, image):
        """检测直线"""
        image_tensor, processed_image = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            y_norm = output.item()
            line_y = int(y_norm * 256)  # 转换为像素坐标
        
        return line_y, processed_image
    
    def visualize(self, image, save_path=None):
        """可视化检测结果"""
        line_y, processed_image = self.detect(image)
        
        # 显示结果
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原始图像或预处理图像
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            display_img = cv2.resize(display_img, (256, 256))
            axes[0].imshow(display_img)
        else:
            display_img = np.array(image.resize((256, 256)).convert('RGB'))
            axes[0].imshow(display_img)
        
        axes[0].axhline(y=line_y, color='red', linewidth=3, linestyle='--', 
                       label=f'Detected: y={line_y}')
        axes[0].set_title('Original Image')
        axes[0].legend(loc='upper right')
        axes[0].axis('off')
        
        # 预处理图像
        axes[1].imshow(processed_image, cmap='gray')
        axes[1].axhline(y=line_y, color='yellow', linewidth=2)
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return line_y

# ==================== 5. 主函数：直线检测 ====================
def main_line_detection():
    print("="*60)
    print("Horizontal Line Detection")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = HorizontalLineDataset(num_samples=8000, train=True)
    val_dataset = HorizontalLineDataset(num_samples=2000, train=False)
    
    # 创建数据加载器
    batch_size = 32 if device == 'cuda' else 16
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    model = LineDetectionCNN()
    
    # 创建训练器
    trainer = LineDetectionTrainer(model, device)
    
    # 开始训练
    save_dir = './line_detection_results'
    train_losses, val_losses, train_maes, val_maes = trainer.train(
        train_loader, val_loader, epochs=50, save_dir=save_dir
    )
    
    # 测试模型
    print("\n" + "="*60)
    print("Testing Model")
    print("="*60)
    
    test_dataset = HorizontalLineDataset(num_samples=100, train=False)
    detector = LineDetector(os.path.join(save_dir, 'best_model.pth'))
    
    errors = []
    
    for i in range(5):  # 测试5个样本
        img, _, true_y = test_dataset[i]
        
        # 转换为numpy用于可视化
        img_np = img.squeeze().numpy()
        
        # 检测直线
        pred_y = detector.visualize(img_np, save_path=os.path.join(save_dir, f'test_result_{i}.png'))
        
        # 计算误差
        error = abs(pred_y - true_y)
        errors.append(error)
        print(f"Test {i+1}: True y={true_y}, Predicted y={pred_y}, Error={error}px")
    
    # 完整测试
    print("\nFull test on 100 samples:")
    errors_full = []
    for i in range(100):
        img, _, true_y = test_dataset[i]
        img_np = img.squeeze().numpy()
        pred_y, _ = detector.detect(img_np)
        error = abs(pred_y - true_y)
        errors_full.append(error)
    
    errors_full = np.array(errors_full)
    print(f"Mean Absolute Error: {errors_full.mean():.2f}px")
    print(f"Standard Deviation: {errors_full.std():.2f}px")
    print(f"Max Error: {errors_full.max():.2f}px")
    
    # 绘制误差分布
    plt.figure(figsize=(10, 6))
    plt.hist(errors_full, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(errors_full.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {errors_full.mean():.2f}px')
    plt.xlabel('Absolute Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution - Line Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nLine detection training and testing completed!")
    print(f"Results saved in: {save_dir}")