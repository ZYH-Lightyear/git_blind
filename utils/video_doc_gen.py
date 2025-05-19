import os
from utils.video_understanding.qwen2 import Qwen2VideoUnderstand
from utils.video_understanding.qwen2_5_api import Qwen2_5VideoUnderstand_API
from utils.tools import load_prompt
from typing import Dict, List

def get_sorted_files(directory):
    # Get absolute paths for all files in directory
    file_paths = [os.path.join(os.path.abspath(directory), file) 
                 for file in os.listdir(directory)]
    # Sort paths
    return sorted(file_paths)

def collect_keyframe_paths(video_path) -> Dict[str, List[str]]:
    """获取指定视频的关键帧图像路径
    Args:
        base_dir: 关键帧根目录 (result_keyframes)
        video_name: 视频名称
    Returns:
        Dict[str, List[str]]: {
            "scene_name": [frame_path1, frame_path2, ...]
        }
    """
    
    scenes = {}
    
    # 获取场景文件夹
    scene_folders = sorted([d for d in os.listdir(video_path) 
                          if os.path.isdir(os.path.join(video_path, d))])
    
    # 处理每个场景文件夹中的关键帧
    for scene in scene_folders:
        scene_path = os.path.join(video_path, scene)
        frame_paths = sorted([
            'file://' + os.path.abspath(os.path.join(scene_path, f))
            for f in os.listdir(scene_path)
            if f.endswith('.jpg')
        ])
        scenes[scene] = frame_paths
    
    return scenes


def get_video_doc(video_path, seg_output_dir, keyframes_path, use_prompt=True, whole_video_caption=None,
                  understand_model=None):
    video_segments_paths = get_sorted_files(seg_output_dir)
    print('共分割出%d个视频片段' % len(video_segments_paths))

    scenes = collect_keyframe_paths(keyframes_path)

    if understand_model is None:
        understand_model = Qwen2_5VideoUnderstand_API()
        understand_model.init_model()
        
    print('开始生成视频描述文档')
    
    video_captions = {}
    if use_prompt:
        prompt = load_prompt('./prompt/video_understand.txt')
        
        # # 使用Qwen2.5本地模型
        # for (scene_name, frames), video_segment in zip(scenes.items(), video_segments_paths):
        #     print(f"\nScene: {scene_name}")
        #     print(f"Frame count: {len(frames)}")
            
        #     result = understand_model.inference(frames, prompt, whole_video_caption)
        #     print(result)
        #     video_captions[video_segment] = result
        
        # 使用Qwen2.5-72B API
        for (scene_name, frames), video_segment in zip(scenes.items(), video_segments_paths):
            print(f"\nScene: {scene_name}")
            print(f"Frame count: {len(frames)}")
            
            result = understand_model.inference(os.path.join(keyframes_path, scene_name), prompt, whole_video_caption="")
            print(result)
            video_captions[video_segment] = result
    
    else:
        video_captions = {path: "" for path in video_segments_paths}

    return video_captions

if __name__ == "__main__":
    name = 'Weapon_23s'
    result_dir = f'./results/{name}'
    seg_output_dir = f'./test_seg/{name}'
    video_path = f'./test_video/mutilclips/{name}.mp4'
    keyframes_path = f'./result_keyframes/{name}/'
    
    doc = get_video_doc(
                video_path=video_path, 
                seg_output_dir=seg_output_dir, 
                keyframes_path=keyframes_path,
                use_prompt=True
                )