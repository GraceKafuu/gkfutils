import cv2
import time
import queue
import logging
import multiprocessing as mp
from multiprocessing import Queue, Event
from typing import Optional, Tuple
import numpy as np
import subprocess
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProducer:
    """图片生产者进程，模拟相机产生图片"""
    
    def __init__(self, image_queue, stop_event, frame_rate):
        self.image_queue = image_queue
        self.stop_event = stop_event
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate
        self.frame_count = 0
        
    def load_image(self, image_path: str) -> np.ndarray:
        """加载图片并确保为3通道BGR格式"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图片: {image_path}")
        
        # 确保图片为3通道BGR格式
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        return img
    
    def run(self, image_path: str):
        """主循环，按指定帧率生产图片"""
        logger.info(f"图片生产者启动，帧率: {self.frame_rate} FPS")
        
        base_image = self.load_image(image_path)
        last_time = time.time()
        
        try:
            while not self.stop_event.is_set():
                current_time = time.time()
                elapsed = current_time - last_time
                
                # 在图片上添加时间戳和帧号用于调试延迟
                display_image = base_image.copy()
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(display_image, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.putText(display_image, f"Time: {timestamp}", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                # 非阻塞方式放入队列，如果队列满则丢弃最旧的帧
                if self.image_queue.full():
                    try:
                        self.image_queue.get_nowait()  # 丢弃最旧的帧
                    except queue.Empty:
                        pass
                
                try:
                    self.image_queue.put_nowait((display_image, self.frame_count, current_time))
                    self.frame_count += 1
                except queue.Full:
                    logger.warning("队列已满，丢弃帧")
                
                # 精确控制帧率
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_time = time.time()
                
        except Exception as e:
            logger.error(f"图片生产者异常: {e}")
        finally:
            logger.info("图片生产者退出")


class RTSPStreamer:
    """RTSP推流器，使用FFmpeg进行高效编码和推流"""
    
    def __init__(self, rtsp_url: str, width: int, height: int, fps: int = 30):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_process = None
        self.is_running = False
        
    def start(self):
        """启动FFmpeg推流进程"""
        try:
            # FFmpeg命令参数，针对低延迟优化
            command = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-f', 'rawvideo',  # 输入格式
                '-vcodec', 'rawvideo',  # 输入编解码器
                '-pix_fmt', 'bgr24',  # 像素格式
                '-s', f'{self.width}x{self.height}',  # 分辨率
                '-r', str(self.fps),  # 帧率
                '-i', '-',  # 从标准输入读取
                '-c:v', 'libx264',  # 视频编码器
                '-preset', 'ultrafast',  # 编码速度预设（最快）
                '-tune', 'zerolatency',  # 零延迟优化
                '-pix_fmt', 'yuv420p',  # 输出像素格式
                '-f', 'rtsp',  # 输出格式
                '-rtsp_transport', 'tcp',  # 使用TCP传输（更稳定）
                self.rtsp_url  # RTSP服务器地址
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0  # 无缓冲
            )
            self.is_running = True
            logger.info(f"RTSP推流器启动: {self.rtsp_url}")
            
        except Exception as e:
            logger.error(f"启动RTSP推流器失败: {e}")
            self.is_running = False
    
    def push_frame(self, frame: np.ndarray):
        """推送一帧到RTSP流"""
        if self.is_running and self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(frame.tobytes())
            except Exception as e:
                logger.error(f"推送帧失败: {e}")
                self.is_running = False
    
    def stop(self):
        """停止推流器"""
        self.is_running = False
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"停止推流器时出错: {e}")
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
        logger.info("RTSP推流器停止")


class ImageProcessor:
    """图片处理器，处理图片并推流"""
    
    def __init__(self, image_queue, stop_event, rtsp_url,  target_width: int = 1700, target_height: int = 660):
        self.image_queue = image_queue
        self.stop_event = stop_event
        self.rtsp_url = rtsp_url
        self.target_width = target_width
        self.target_height = target_height
        self.streamer = None
        self.processing_times = []
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图片，模拟30-50ms的处理时间
        返回处理后的图片
        """
        start_time = time.time()
        
        try:
            # 1. 调整图片尺寸
            # processed_frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # 2. 模拟图像处理操作（可根据需要调整）
            # 高斯模糊
            processed_frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # 边缘检测
            edges = cv2.Canny(processed_frame, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # 融合原图和边缘检测结果
            processed_frame = cv2.addWeighted(processed_frame, 0.7, edges_colored, 0.3, 0)
            
            # 3. 添加处理信息
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            # 保持最近100帧的统计数据
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            avg_processing_time = np.mean(self.processing_times)
            cv2.putText(processed_frame, f"Proc: {processing_time:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Avg: {avg_processing_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Frame: {self.frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 4. 控制处理时间在30-50ms范围内
            current_processing_time = (time.time() - start_time) * 1000
            target_processing_time = np.random.uniform(30, 50)
            
            if current_processing_time < target_processing_time:
                # 如果处理太快，等待以达到目标处理时间
                sleep_time = (target_processing_time - current_processing_time) / 1000.0
                time.sleep(max(0, sleep_time))
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
            # 返回一个默认的处理后图片
            return np.ones((self.target_height, self.target_width, 3), dtype=np.uint8) * 128
    
    def run(self):
        """主处理循环"""
        logger.info("图片处理器启动")
        
        # 初始化RTSP推流器
        self.streamer = RTSPStreamer(self.rtsp_url, self.target_width, self.target_height)
        self.streamer.start()
        
        last_stat_time = time.time()
        stat_interval = 5  # 每5秒输出一次统计信息
        
        try:
            while not self.stop_event.is_set() or not self.image_queue.empty():
                try:
                    # 非阻塞获取图片，超时时间短
                    frame, frame_num, produce_time = self.image_queue.get(timeout=0.1)
                    
                    # 计算生产到处理的延迟
                    produce_delay = (time.time() - produce_time) * 1000
                    
                    # 处理图片
                    processed_frame = self.process_frame(frame)
                    
                    # 推流
                    if self.streamer and self.streamer.is_running:
                        self.streamer.push_frame(processed_frame)
                    
                    self.frame_count += 1
                    
                    # 定期输出统计信息
                    current_time = time.time()
                    if current_time - last_stat_time >= stat_interval:
                        if self.processing_times:
                            avg_time = np.mean(self.processing_times)
                            max_time = np.max(self.processing_times)
                            min_time = np.min(self.processing_times)
                            queue_size = self.image_queue.qsize()
                            
                            logger.info(
                                f"处理统计 - 平均: {avg_time:.1f}ms, "
                                f"最大: {max_time:.1f}ms, 最小: {min_time:.1f}ms, "
                                f"队列大小: {queue_size}, 总帧数: {self.frame_count}"
                            )
                        last_stat_time = current_time
                        
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"处理循环中出错: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"图片处理器异常: {e}")
        finally:
            if self.streamer:
                self.streamer.stop()
            logger.info("图片处理器退出")


class VideoProcessingSystem:
    """视频处理系统主控制器"""
    
    def __init__(self, image_path: str, rtsp_url: str = "rtsp://localhost:8554/live.stream", 
                 frame_rate: int = 30, queue_size: int = 10):
        self.image_path = image_path
        self.rtsp_url = rtsp_url
        self.frame_rate = frame_rate
        self.queue_size = queue_size
        
        self.stop_event = Event()
        self.image_queue = Queue(maxsize=queue_size)
        
        self.producer_process = None
        self.processor_process = None
        
    def start(self):
        """启动系统"""
        logger.info("启动视频处理系统")
        
        # 创建并启动生产者进程
        producer = ImageProducer(self.image_queue, self.stop_event, self.frame_rate)
        self.producer_process = mp.Process(
            target=producer.run, 
            args=(self.image_path,),
            name="ImageProducer"
        )
        self.producer_process.daemon = True
        self.producer_process.start()
        
        # 创建并启动处理器进程
        processor = ImageProcessor(self.image_queue, self.stop_event, self.rtsp_url)
        self.processor_process = mp.Process(
            target=processor.run,
            name="ImageProcessor"
        )
        self.processor_process.daemon = True
        self.processor_process.start()
        
        logger.info("视频处理系统启动完成")
    
    def stop(self):
        """停止系统"""
        logger.info("停止视频处理系统")
        self.stop_event.set()
        
        # 等待进程结束
        if self.producer_process:
            self.producer_process.join(timeout=5)
            if self.producer_process.is_alive():
                self.producer_process.terminate()
        
        if self.processor_process:
            self.processor_process.join(timeout=5)
            if self.processor_process.is_alive():
                self.processor_process.terminate()
        
        # 清空队列
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("视频处理系统已停止")
    
    def run(self):
        """运行系统直到被中断"""
        try:
            self.start()
            # 保持主线程运行，直到收到中断信号
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            self.stop()

def main():
    """主函数"""
    # 配置参数
    IMAGE_PATH = r"G:\Gosion\data\006.Belt_Torn_Det\data\seg\3d_seg\images\20250915142244_19_28.jpg"  # 替换为你的图片路径
    RTSP_URL = "rtsp://127.0.0.1:8554/test"  # RTSP服务器地址
    FRAME_RATE = 30  # 帧率
    QUEUE_SIZE = 1   # 队列大小（保持较小以减少延迟）
    
    # 创建并运行系统
    system = VideoProcessingSystem(
        image_path=IMAGE_PATH,
        rtsp_url=RTSP_URL,
        frame_rate=FRAME_RATE,
        queue_size=QUEUE_SIZE
    )
    
    system.run()

if __name__ == "__main__":
    main()