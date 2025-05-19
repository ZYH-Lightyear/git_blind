import os
from openai import OpenAI
import base64
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from utils.tools import load_prompt
import json
import cv2

class Qwen2_5VideoUnderstand_API:
    def __init__(self):
        """初始化类"""
        self.client = None
        self.model_name = "qwen-vl-max"
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.system_prompt_path = "./prompt/video_understand.txt"
        self.whole_video_understand_path = "./prompt/whole_video_understand.txt"
        
    def init_model(self) -> None:
        """初始化OpenAI客户端"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.system_prompt = load_prompt(self.system_prompt_path)
        return self
    
    @staticmethod
    def encode_video(video_path: str) -> str:
        """视频文件Base64编码"""
        content_list = []
        with open(video_path, "rb") as video_file:
            base64_video = base64.b64encode(video_file.read()).decode("utf-8")
            content_list.append({
                "type": "video_url",
                "video_url": {
                    "url": f"data:;base64,{base64_video}"
                }
            })
            # print(base64_video)
        return content_list

    @staticmethod
    def img_to_base64_str(img_path: str) -> str:
        """图片文件Base64编码"""
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')    
        

    def get_base64_images(self, image_folder: str) -> List[dict]:
        """处理文件夹中的图片并转换为base64格式
        Args:
            image_folder: 图片文件夹路径
            custom_prompt: 自定义提示词
        Returns:
            List[dict]: 包含base64编码图片和提示词的消息列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png']
        content_list = []
        
        folder = Path(image_folder).resolve()
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in image_extensions:
                with open(img_path, 'rb') as image_file:
                    # 读取图片
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"Warning: Unable to read image {img_path}")
                        continue
                    
                    # 获取原始尺寸
                    height, width = img.shape[:2]
                    
                    # 计算缩放比例，保持宽高比
                    if height > width:  # 竖屏图片
                        new_height = 480
                        new_width = int(width * (new_height / height))
                    else:  # 横屏图片
                        new_width = 480
                        new_height = int(height * (new_width / width))
                    
                    # 确保宽高是4的倍数（可选，有些处理可能需要）
                    new_width = new_width - (new_width % 4)
                    new_height = new_height - (new_height % 4)
                    
                    # 缩放图片
                    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # 转换为base64
                    _, buffer = cv2.imencode('.jpg', resized_img)
                    base64_img = base64.b64encode(buffer).decode('utf-8')
                    # base64_img = base64.b64encode(image_file.read()).decode('utf-8')
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    })
        # print(content_list)
        return content_list
    
    def inference(self, image_folder: str, prompt: Optional[str] = None, whole_video_caption=None) -> str:
        """分析关键帧图片并生成描述
        Args:
            image_folder: 关键帧图片文件夹路径
            custom_prompt: 自定义提示词
        Returns:
            str: 生成的场景描述
        """
        if self.client is None:
            self.init_model()

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}]
            },
            {
                "role": "user",
                "content": self.get_base64_images(image_folder)
            }
        ]
        
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": prompt}]
        #     },
        #     {
        #         "role": "user",
        #         "content": self.get_base64_images(image_folder)
        #     }
        # ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Error analyzing keyframes: {str(e)}")
            return ""

    def extract_frames_fps(self, video_path: str, fps: int = 1) -> List[np.ndarray]:
        """按指定FPS提取视频帧
        
        Args:
            video_path: 视频文件路径
            fps: 期望的帧率，默认1fps
            
        Returns:
            List[np.ndarray]: 提取的帧列表
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 获取视频信息
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(orig_fps / fps)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # 获取原始尺寸
                    height, width = frame.shape[:2]
                    
                    # 计算缩放比例，保持宽高比
                    if height > width:  # 竖屏视频
                        new_height = 480
                        new_width = int(width * (new_height / height))
                    else:  # 横屏视频
                        new_width = 480
                        new_height = int(height * (new_width / width))
                    
                    new_width = new_width - (new_width % 4)
                    new_height = new_height - (new_height % 4)
                    
                    # 缩放图片
                    resized_frame = cv2.resize(frame, (new_width, new_height), 
                                            interpolation=cv2.INTER_AREA)
                    
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    
                frame_count += 1
                
            cap.release()
            return frames
            
        except Exception as e:
            print(f"提取视频帧时出错: {str(e)}")
            return []

    def frames_to_base64(self, frames: List[np.ndarray]) -> List[Dict]:
        """将帧列表转换为base64编码的消息列表
        
        Args:
            frames: numpy数组格式的帧列表
            
        Returns:
            List[Dict]: 包含base64编码的消息列表
        """
        content_list = []
        try:
            for frame in frames:
                # 将numpy数组编码为jpg
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                base64_str = base64.b64encode(buffer).decode('utf-8')
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}"
                    }
                })
            return content_list
        except Exception as e:
            print(f"转换帧到base64时出错: {str(e)}")
            return []

    def inference_video_frames(self, video_path: str, fps: int = 1) -> str:
        """使用逐帧分析方式推理视频内容
        
        Args:
            video_path: 视频文件路径
            prompt: 自定义提示词
            fps: 提取帧的帧率
            
        Returns:
            str: 推理结果
        """
        if self.client is None:
            self.init_model()
            
        # 提取帧
        frames = self.extract_frames_fps(video_path, fps)
        if not frames:
            return "Failed to extract video frames"
            
        # 转换为base64格式
        content_list = self.frames_to_base64(frames)
        if not content_list:
            return "Failed to convert frames to base64"
                
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": load_prompt(self.whole_video_understand_path)}]
            },
            {
                "role": "user",
                "content": content_list
            }
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"视频帧推理时出错: {str(e)}")
            return ""

# Usage example
if __name__ == "__main__":
    # Create instance
    model = Qwen2_5VideoUnderstand_API()
    
    # Optional: load custom prompts
    # model.load_prompt("path/to/prompt_config.json")
    
    # Initialize model
    model.init_model()
    
    # Analyze keyframes
    result = model.inference("/hpc2hdd/home/yzhang679/codes/MMAudio/Weapon_23s-Scene-010")
    print(result)
    
    # 测试1FPS提取帧并进行推理
    video_path = "/path/to/your/video.mp4"
    result = model.inference_video_frames(
        video_path,
        prompt="请描述视频中的主要事件和变化，注意时序信息",
        fps=1
    )
    print(result)