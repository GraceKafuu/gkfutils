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




# ==================== 1. 曲线检测数据集 ====================
class CurveDataset(Dataset):
    """生成带随机曲线的图像数据集"""
    def __init__(self, num_samples=10000, img_size=(256, 256), 
                 curve_width_range=(1, 3), train=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.curve_width_range = curve_width_range
        self.train = train
        self.num_points = 256  # 输出256个点
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 创建背景
        bg_type = np.random.choice(['uniform', 'gradient', 'noise'])
        
        if bg_type == 'uniform':
            bg_value = np.random.uniform(0.3, 0.7)
            img = np.ones(self.img_size) * bg_value
        elif bg_type == 'gradient':
            gradient = np.linspace(0.2, 0.8, self.img_size[0]).reshape(-1, 1)
            img = np.tile(gradient, (1, self.img_size[1]))
        else:  # noise
            img = np.random.randn(*self.img_size) * 0.15 + 0.5
        
        # 随机生成曲线参数
        # 使用二次函数: y = a*(x - center_x)^2 + b
        center_x = np.random.uniform(0.3, 0.7) * self.img_size[1]
        
        # 随机选择开口方向
        if np.random.rand() > 0.5:
            a = np.random.uniform(0.0001, 0.001)  # 开口向上
        else:
            a = np.random.uniform(-0.001, -0.0001)  # 开口向下
        
        b = np.random.randint(50, self.img_size[0] - 50)  # 顶点y坐标
        
        # 生成曲线点
        x_coords = np.linspace(0, self.img_size[1] - 1, self.num_points)
        y_coords = []
        
        for x in x_coords:
            y = a * (x - center_x) ** 2 + b
            # 添加一些随机扰动使曲线更自然
            if self.train:
                y += np.random.randn() * 2
            y = np.clip(y, 0, self.img_size[0] - 1)
            y_coords.append(y)
        
        y_coords = np.array(y_coords)
        
        # 在图像上绘制曲线
        curve_width = np.random.randint(*self.curve_width_range)
        half_width = curve_width // 2
        
        # 确定曲线对比度
        if np.random.rand() > 0.5:
            curve_intensity = min(1.0, img.mean() + np.random.uniform(0.4, 0.9))
        else:
            curve_intensity = max(0.0, img.mean() - np.random.uniform(0.4, 0.9))
        
        # 绘制曲线
        for i in range(len(x_coords) - 1):
            x0, y0 = int(x_coords[i]), int(y_coords[i])
            x1, y1 = int(x_coords[i+1]), int(y_coords[i+1])
            
            # 绘制线段
            cv2.line(img, (x0, y0), (x1, y1), curve_intensity, curve_width)
            
            # 在控制点处加粗
            if i % 50 == 0:
                radius = curve_width + 1
                cv2.circle(img, (x0, y0), radius, curve_intensity, -1)
        
        # 添加噪声
        noise = np.random.randn(*self.img_size) * 0.1
        img += noise
        
        # 添加干扰（训练时）
        if self.train:
            # 随机点
            num_dots = np.random.randint(10, 25)
            for _ in range(num_dots):
                px = np.random.randint(0, self.img_size[1])
                py = np.random.randint(0, self.img_size[0])
                radius = np.random.randint(1, 4)
                intensity = np.random.uniform(0.3, 0.7)
                py_start = max(0, py - radius)
                py_end = min(self.img_size[0], py + radius + 1)
                px_start = max(0, px - radius)
                px_end = min(self.img_size[1], px + radius + 1)
                img[py_start:py_end, px_start:px_end] += intensity
            
            # 偶尔添加直线干扰
            if np.random.rand() < 0.15:
                line_y = np.random.randint(30, self.img_size[0] - 30)
                width = np.random.randint(1, 3)
                intensity = np.random.uniform(0.3, 0.6)
                img[line_y:line_y+width, :] += intensity
        
        # 裁剪并归一化
        img = np.clip(img, 0, 1)
        
        # 标准化
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # 转换为张量
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        
        # 归一化标签到[0, 1]
        labels_normalized = y_coords / self.img_size[0]
        labels_tensor = torch.FloatTensor(labels_normalized)
        
        # 保存原始坐标用于评估
        original_coords = y_coords.copy()
        
        return img_tensor, labels_tensor, original_coords


# ==================== 2. 曲线检测网络 ====================
class CurveDetectionCNN(nn.Module):
    """曲线检测CNN - 输出256个y坐标值"""
    def __init__(self, num_points=256):
        super().__init__()
        self.num_points = num_points
        
        # 特征提取部分 - 使用全卷积架构
        self.encoder = nn.Sequential(
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
            
            # 空间金字塔池化
            nn.AdaptiveAvgPool2d((8, 8)),  # 512x8x8
        )
        
        # 解码器 - 输出256个点
        self.decoder = nn.Sequential(
            # 将空间特征转换为序列
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 全局池化得到256维特征
            nn.AdaptiveAvgPool2d((1, 256)),  # 256x1x256
            
            # 调整维度
            nn.Flatten(start_dim=1, end_dim=2),  # 256x256
            
            # 全连接层细化
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 输出层
            nn.Linear(512, num_points),
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
        features = self.encoder(x)
        output = self.decoder(features)
        return output

# ==================== 3. 曲线检测训练器 ====================
class CurveDetectionTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 使用Smooth L1 Loss
        self.criterion = nn.SmoothL1Loss(beta=0.1)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.best_val_mae = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测结果
            all_predictions.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算MAE
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        mae = mean_absolute_error(all_labels.flatten(), all_predictions.flatten())
        
        return total_loss / len(dataloader), mae
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_original_coords = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for images, labels, original_coords in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 收集结果
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_original_coords.append(original_coords.numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_original_coords = np.concatenate(all_original_coords, axis=0)
        
        # 归一化空间的MAE
        mae_norm = mean_absolute_error(all_labels.flatten(), all_predictions.flatten())
        
        # 像素空间的MAE
        predictions_pixel = all_predictions * 256  # 转换为像素坐标
        mae_pixel = mean_absolute_error(all_original_coords.flatten(), predictions_pixel.flatten())
        
        # 逐点平均MAE
        pointwise_mae = np.mean(np.abs(all_original_coords - predictions_pixel), axis=0)
        
        return total_loss / len(dataloader), mae_norm, mae_pixel, pointwise_mae
    
    def train(self, train_loader, val_loader, epochs=50, save_dir='./curve_detection'):
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
            val_loss, val_mae_norm, val_mae_pixel, pointwise_mae = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae_pixel)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 打印结果
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}")
            print(f"  Val Loss: {val_loss:.6f}, Val MAE: {val_mae_pixel:.2f}px")
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
                
                # 绘制点误差分布
                self.plot_pointwise_error(pointwise_mae, epoch+1, save_dir)
        
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
    
    def plot_pointwise_error(self, pointwise_mae, epoch, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(pointwise_mae, linewidth=2)
        plt.axhline(y=pointwise_mae.mean(), color='r', linestyle='--', 
                   label=f'Mean: {pointwise_mae.mean():.2f}px')
        plt.xlabel('Point Index (x coordinate)')
        plt.ylabel('MAE (pixels)')
        plt.title(f'Point-wise Error Distribution - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'pointwise_error_epoch_{epoch}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

# ==================== 4. 曲线检测器 ====================
class CurveDetector:
    """曲线检测器"""
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_points = 256
        self.model = CurveDetectionCNN(num_points=self.num_points)
        
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
        """检测曲线"""
        image_tensor, processed_image = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            curve_points_norm = output.squeeze().cpu().numpy()
            curve_points = curve_points_norm * 256  # 转换为像素坐标
        
        return curve_points, processed_image
    
    def visualize(self, image, save_path=None):
        """可视化检测结果"""
        curve_points, processed_image = self.detect(image)
        
        # 生成x坐标
        x_coords = np.linspace(0, 255, self.num_points)
        
        # 显示结果
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 原始图像
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
        
        axes[0].plot(x_coords, curve_points, 'r-', linewidth=2, alpha=0.8)
        axes[0].scatter(x_coords[::10], curve_points[::10], c='yellow', s=30, 
                       edgecolors='black', zorder=5)
        axes[0].set_title('Original Image with Detected Curve')
        axes[0].axis('off')
        
        # 预处理图像
        axes[1].imshow(processed_image, cmap='gray')
        axes[1].plot(x_coords, curve_points, 'y-', linewidth=2, alpha=0.8)
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        # 曲线图
        axes[2].plot(x_coords, curve_points, 'b-', linewidth=3, label='Detected Curve')
        axes[2].set_xlabel('X Coordinate')
        axes[2].set_ylabel('Y Coordinate')
        axes[2].set_title('Detected Curve Points')
        axes[2].set_ylim(0, 256)
        axes[2].invert_yaxis()  # 图像坐标系y轴向下
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return curve_points

# ==================== 5. 主函数：曲线检测 ====================
def main_curve_detection():
    print("="*60)
    print("Curve Detection")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_dataset = CurveDataset(num_samples=8000, train=True)
    val_dataset = CurveDataset(num_samples=2000, train=False)
    
    # 创建数据加载器
    batch_size = 16 if device == 'cuda' else 8  # 曲线检测需要更多内存
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
    model = CurveDetectionCNN(num_points=256)
    
    # 创建训练器
    trainer = CurveDetectionTrainer(model, device)
    
    # 开始训练
    save_dir = './curve_detection_results'
    train_losses, val_losses, train_maes, val_maes = trainer.train(
        train_loader, val_loader, epochs=50, save_dir=save_dir
    )
    
    # 测试模型
    print("\n" + "="*60)
    print("Testing Model")
    print("="*60)
    
    test_dataset = CurveDataset(num_samples=100, train=False)
    detector = CurveDetector(os.path.join(save_dir, 'best_model.pth'))
    
    point_errors = []
    
    for i in range(3):  # 测试3个样本
        img, _, true_points = test_dataset[i]
        
        # 转换为numpy用于可视化
        img_np = img.squeeze().numpy()
        
        # 检测曲线
        pred_points = detector.visualize(img_np, save_path=os.path.join(save_dir, f'curve_test_{i}.png'))
        
        # 计算逐点误差
        point_error = np.abs(pred_points - true_points)
        point_errors.append(point_error)
        
        # 计算平均误差
        avg_error = point_error.mean()
        max_error = point_error.max()
        print(f"Test {i+1}: Avg Error={avg_error:.2f}px, Max Error={max_error:.2f}px")
    
    # 完整测试
    print("\nFull test on 50 samples:")
    all_errors = []
    for i in range(50):
        img, _, true_points = test_dataset[i]
        img_np = img.squeeze().numpy()
        pred_points, _ = detector.detect(img_np)
        error = np.abs(pred_points - true_points)
        all_errors.append(error)
    
    all_errors = np.concatenate(all_errors)
    print(f"Overall Mean Absolute Error: {all_errors.mean():.2f}px")
    print(f"Overall Standard Deviation: {all_errors.std():.2f}px")
    print(f"Overall Max Error: {all_errors.max():.2f}px")
    
    # 绘制误差分布
    plt.figure(figsize=(12, 5))
    
    # 误差直方图
    plt.subplot(1, 2, 1)
    plt.hist(all_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(all_errors.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {all_errors.mean():.2f}px')
    plt.xlabel('Point-wise Error (pixels)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution - Curve Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 逐点平均误差
    plt.subplot(1, 2, 2)
    pointwise_mean_error = np.array([err.mean() for err in point_errors])
    plt.plot(range(len(pointwise_mean_error)), pointwise_mean_error, 'bo-', linewidth=2)
    plt.axhline(y=pointwise_mean_error.mean(), color='r', linestyle='--', 
               label=f'Mean: {pointwise_mean_error.mean():.2f}px')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Mean Error (pixels)')
    plt.title('Sample-wise Mean Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nCurve detection training and testing completed!")
    print(f"Results saved in: {save_dir}")


# ==================== 6. 主执行函数 ====================
if __name__ == "__main__":
    main_curve_detection()