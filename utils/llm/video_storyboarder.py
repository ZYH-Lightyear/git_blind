import os
from openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from pathlib import Path
from utils.tools import load_prompt, get_sorted_files, merge_two_videos
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from utils.segment_video import segment_video
import json
from .base import BaseLLM
from utils.label_rag.llamaindex_rag import LabelRAG
from scenedetect import detect, ContentDetector, split_video_ffmpeg
import os
import ffmpeg
from moviepy import VideoFileClip
import math
import shutil

class Video_Storyboard(BaseLLM):
    def __init__(
        self, model_name: str = "qwen-plus",
        api_base: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None,
        prompt_path: str = "./prompt/storyboarder.txt"
    ) -> None:
        super().__init__(model_name)
        self.api_base = api_base
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.client = None
        self.prompt_path = prompt_path
        self.system_prompt = load_prompt(prompt_path)
        # self._setup_tool()
        self.init_model()   

    def init_model(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def get_completion(self, messages: List[Dict[str, str]],format="text"):
        """Get conversation completion"""
        kwargs = {"model": self.model_name, "messages": messages, "response_format": {"type": format}}
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.model_dump()
    

    def generate(
        self, 
        caption1: str,
        caption2: str,
        response_format: str = "json_object",
        **kwargs
    ) -> str:
        """
        Generate response
        Args:
            prompt: User prompt
            response_format: Return format
            **kwargs: Other parameters
        Returns:
            str: Model-generated response
        """
        if self.client is None:
            self.init_model()
        
        try:
            self.system_prompt = load_prompt(self.prompt_path)

            messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Last Video Caption: {caption1}, \n Current Video Caption: {caption2}"}
                ]
            
            first_response = self.get_completion(messages, format="json_object")
            assistant_output = first_response['choices'][0]['message']
            # print(assistant_output['content'])
            
            parse_res = self.parse_json(assistant_output['content'])
            merge = parse_res['merge']

            caption = parse_res['caption']
            print(f'caption: {caption}')

            return caption, merge

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return False
    
    def _split_by_duration(self, video_path: str, output_dir: str, segment_duration: int = 10) -> list:
        """Split video by fixed duration"""
        video = VideoFileClip(video_path)
        total_duration = video.duration
        scenes = []
        
        try:
            # Calculate number of segments needed
            num_segments = math.ceil(total_duration / segment_duration)
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, total_duration)
                
                # Extract segment
                clip = video.subclipped(start_time, end_time)
                output_path = os.path.join(output_dir, f"scene_{i:03d}.mp4")
                clip.write_videofile(output_path)
                clip.close()
                
                scenes.append((start_time, end_time))
                
        finally:
            video.close()
            
        return scenes
    
    def _process_long_segment(self, scenes: list, input_path: str, output_dir: str) -> list:
        """Process long video segments, further split segments longer than 12 seconds"""
        new_scenes = []
        video = VideoFileClip(input_path)
        
        try:
            for i, (start_time, end_time) in enumerate(scenes):
                start_time = start_time.get_seconds()
                end_time = end_time.get_seconds()
                duration = end_time - start_time
                if duration > 15:
                    # For segments that are too long, split by 10s
                    num_sub_segments = math.ceil(duration / 10)
                    for j in range(num_sub_segments):
                        sub_start = start_time + j * 10
                        sub_end = min(sub_start + 10, end_time)
                        
                        # Extract and save sub-segment
                        sub_clip = video.subclipped(sub_start, sub_end)
                        output_path = os.path.join(output_dir, f"scene_{i:03d}_{j:03d}.mp4")
                        sub_clip.write_videofile(output_path)
                        # sub_clip.close()
                        
                        new_scenes.append((sub_start, sub_end))
                else:
                    # For segments of normal length, keep as is
                    output_path = os.path.join(output_dir, f"scene_{i:03d}.mp4")
                    clip = video.subclipped(start_time, end_time)
                    clip.write_videofile(output_path)
                    # clip.close()
                    new_scenes.append((start_time, end_time))
        finally:
            video.close()
            
        return new_scenes

    def _handle_short_segments(self, scenes: list) -> list:
        """Handle short video segments (less than 0.5 seconds)"""
        if not scenes:
            return scenes
            
        processed_scenes = []
        i = 0
        while i < len(scenes):
            start_time, end_time = scenes[i]
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()
            duration = end_sec - start_sec
            
            if duration < 0.5:  # If segment is too short
                if i == len(scenes) - 1 and processed_scenes:  # If it's the last segment
                    # Merge with previous segment
                    prev_start, prev_end = processed_scenes[-1]
                    processed_scenes[-1] = (prev_start, end_time)
                elif i < len(scenes) - 1:  # If not the last segment
                    # Merge with next segment
                    next_start, next_end = scenes[i + 1]
                    scenes[i + 1] = (start_time, next_end)  # Update time range of next segment
                i += 1
                continue
                
            processed_scenes.append((start_time, end_time))
            i += 1
            
        return processed_scenes

    def segment_video(self, input_path: str, output_dir: str = './video_segments', save=True) -> list:
        """
        Segment video into scenes based on content detection.
        Processing logic:
        1. First detect scenes
        2. Process short segments (<0.5s)
        3. Process long segments (>12s)
        4. If no scenes detected, process by duration
        
        Args:
            input_path (str): Path to input video file
            output_dir (str): Directory to save segmented videos
            save (bool): Whether to save video segments
            
        Returns:
            list: List of scene timestamps (scene_start, scene_end)
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Get video duration
        video = VideoFileClip(input_path)
        duration = video.duration
        # video.close()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect scenes
        scenes = detect(input_path, ContentDetector(), './stats.csv')
        
        if not scenes:  # If no scene changes detected
            print("No scene changes detected, checking duration...")
            if duration > 12:
                print(f"Video duration ({duration}s) > 12s, splitting by 10s intervals")
                return self._split_by_duration(input_path, output_dir)
            else:
                print(f"Video duration ({duration}s) <= 12s, saving as single clip")
                if save:
                    output_path = os.path.join(output_dir, "scene_000.mp4")
                    shutil.copy2(input_path, output_path)
                return [(0, duration)]
        
        # Handle short segments
        print("Checking for short segments (<0.5s)...")
        scenes = self._handle_short_segments(scenes)
        
        # Check if there are long segments that need further splitting
        has_long_segment = any((end.get_seconds() - start.get_seconds()) > 12 for start, end in scenes)
        
        if has_long_segment and save:
            print("Found segments longer than 12s, performing additional splitting...")
            return self._process_long_segment(scenes, input_path, output_dir)
        else:
            # Split according to processed scenes
            print(f"Final number of scenes: {len(scenes)}")
            for scene_start, scene_end in scenes:
                print(f'{scene_start}-{scene_end}')
            
            if save:
                split_video_ffmpeg(
                    input_video_path=input_path,
                    scene_list=scenes,
                    output_dir=output_dir
                )
            
            return scenes

    def parse_json(self, response: str):
        """
        Parse JSON response
        Args:
            response: Model-generated response
        Returns:
            Dict[str, Any]: Parsed response
        """
        return json.loads(response)

    def judge_merge(self, seg_dir, video_captions, whole_video_caption):
        # Get video segment paths
        video_segments_paths = get_sorted_files(seg_dir)
        processed_segments = []  # Track processed segments
        
        if video_segments_paths:
            processed_segments.append(video_segments_paths[0])
            # Content-based merge judgment
            # processed_segments = storyboarder.generate(video_segments_paths=video_segments_paths, video_captions=video_captions,whole_video_caption=whole_video_caption)
            i = 1
            while i < len(video_segments_paths):
                current_path = video_segments_paths[i]
                prev_path = processed_segments[-1]
                new_caption, merge_ = self.generate(video_captions[prev_path], video_captions[current_path])
                merge = True if merge_ == "True" else False
                print(f"Merge: {merge}")
                
                # Determine if merging is needed
                if merge:
                    print(f"Merging segments {prev_path} and {current_path}")
                    # Create path for temporary merged file
                    temp_merged = str(Path(current_path).parent / f"temp_merged_{datetime.now().timestamp()}.mp4")
                    
                    # Merge videos
                    merge_two_videos(prev_path, current_path, temp_merged)
                    
                    # Replace files
                    os.remove(prev_path)  # Delete previous video
                    os.rename(temp_merged, current_path)  # Rename merged video to current video name
                    
                    # Update video captions
                    video_captions[current_path] = new_caption
                    if prev_path in video_captions:
                        del video_captions[prev_path]  # Delete old video caption
                    
                    # Update list of processed segments
                    processed_segments[-1] = current_path
                else:
                    video_captions[current_path] = new_caption
                    processed_segments.append(current_path)
                i += 1
            
            # Update list of processed video segments
            return processed_segments, video_captions

