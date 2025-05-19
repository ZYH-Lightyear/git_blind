import logging
from moviepy import *
from pathlib import Path
from utils.sample_key_frame import process_video_folder
from utils.video_doc_gen import get_video_doc
from utils.llm.audio_generator import AudioPromptOptimizer
import numpy as np
from utils.llm.audio_designer import InitialAudioPromptGen
from utils.video_understanding.qwen2_5_api import Qwen2_5VideoUnderstand_API
from utils.llm.video_storyboarder import Video_Storyboard
from utils.tools import get_sorted_files, get_extended_filename, merge_videos_in_directory, extend_short_video_by_frames, replace_video_audio, merge_two_videos
import gradio as gr
import torch
import torchaudio
from typing import Dict, Any, List
import os
from pydub import AudioSegment
import sys
from tqdm import tqdm

try:
    import mmaudio
except ImportError:
    os.system("pip install -e .")
    import mmaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils
import tempfile

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cuda'
dtype = torch.bfloat16

model: ModelConfig = all_model_cfg['large_44k_v2']
model.download_if_needed()
output_dir = Path('./output/gradio')

setup_eval_logging()

# Create different functional agents for this example
print("Creating Storyboarder...")
storyboarder = Video_Storyboard(
    model_name="qwen-max",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

log.info("Creating Scriptwriter...")

scriptwriter = Qwen2_5VideoUnderstand_API()
scriptwriter.init_model()

log.info("Creating Audio Designer...")
audio_designer = InitialAudioPromptGen(
    model_name="qwen-max",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

log.info("Creating Audio Generator...")
audio_generator =AudioPromptOptimizer(
    model_name="qwen-max",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

def get_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = model.seq_cfg

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')
    
    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, seq_cfg

net = None
feature_utils = None
seq_cfg = None

# net, feature_utils, seq_cfg = get_model()

# Modify video_process to return preview updates
def video_process(video: gr.Video, progress=gr.Progress()):
    video_path = Path(video)
    temp_dir = video_path.parent
    
    seg_dir = temp_dir / 'segments'
    keyframes_dir = temp_dir / 'keyframes'
    clip_results_dir = temp_dir / 'clip_results'

    logging.info(f"Video path: {video}")
    logging.info(f"Temp directory: {str(temp_dir)}")
    logging.info(f"Segments directory: {str(seg_dir)}")
    logging.info(f"Keyframes directory: {str(keyframes_dir)}")
    logging.info(f"Result dir: {str(clip_results_dir)}")

    scenes = storyboarder.segment_video(input_path=video, output_dir=seg_dir)

    results = process_video_folder(input_folder=seg_dir, output_base_dir=keyframes_dir)

    whole_video_caption = scriptwriter.inference_video_frames(video_path=video)
    print(whole_video_caption)
    

    video_captions = {}
    video_captions = get_video_doc(video_path=video, seg_output_dir=seg_dir, keyframes_path=keyframes_dir, use_prompt=True, understand_model=scriptwriter)

    video_segments_paths = get_sorted_files(seg_dir)

    if video_segments_paths:
        video_segments_paths, video_captions = storyboarder.judge_merge(seg_dir, video_captions, whole_video_caption)

    return video_segments_paths, video_captions


def video_to_audio(video: gr.Video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                   cfg_strength: float, duration: float, video_save_path = None, return_segments=False):
    
    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()

    video_path = Path(video)
    temp_dir = video_path.parent
    
    seg_dir = temp_dir / 'segments'
    keyframes_dir = temp_dir / 'keyframes'
    clip_results_dir = temp_dir / 'clip_results'

    logging.info(f"Video path: {video}")
    logging.info(f"Temp directory: {str(temp_dir)}")
    logging.info(f"Segments directory: {str(seg_dir)}")
    logging.info(f"Keyframes directory: {str(keyframes_dir)}")
    logging.info(f"Result dir: {str(clip_results_dir)}")

    video_segments_paths, video_captions = video_process(video)
    
    if video_segments_paths:
        audio_prompts = {}
        
        if os.path.exists(clip_results_dir) == False:
            print("Creating clip results directory")
            os.makedirs(clip_results_dir, exist_ok=True)
        
        processed_segment_paths = []
        for i in range(len(video_segments_paths)):
            logging.info("Initial audio prompt generation...")
            if i == 0:
                audio_prompt = audio_designer.generate(prompt=video_captions[video_segments_paths[i]], last_video_caption="")
            else:
                audio_prompt = audio_designer.generate(prompt=video_captions[video_segments_paths[i]], last_video_caption=video_captions[video_segments_paths[i-1]])
            logging.info(f"Initial audio prompt:\n{audio_prompt}")

            audio_prompt = audio_generator.generate(video_caption=video_captions[video_segments_paths[i]], audio_description=audio_prompt)
            
            parse_prompt = audio_designer.parse_json(audio_prompt)

            audio_prompts[video_segments_paths[i]] = parse_prompt
            
            logging.info(f"Generated audio prompt:\n{audio_prompt}")
            logging.info(f"Video Caption:\n{video_captions[video_segments_paths[i]]}")


            video = VideoFileClip(video_segments_paths[i]) 

            video_duration = video.duration
            print(f"Video length: {video_duration} seconds")
            
            current_video_path = video_segments_paths[i]
            if video.duration < 0.8:
                print(f"Video length less than 0.8 seconds, extending duration...")
                input_video_path = video_segments_paths[i]
                extended_path = get_extended_filename(input_video_path, clip_results_dir)
                extended_path = extend_short_video_by_frames(input_video_path, extended_path)
                
                print(extended_path)
                
                video = VideoFileClip(extended_path)
                video_duration = video.duration
                current_video_path = extended_path
            
            
            audio = layer_audio_gen(
                video_path=current_video_path,
                audio_description=audio_prompts[video_segments_paths[i]],
                duration=video_duration,
                negative_prompt=negative_prompt,
                seed=seed,
                num_steps=num_steps,
                cfg_strength=cfg_strength
            )
            
            if audio is None:
                continue
            
            
            video_info = load_video(current_video_path, video_duration)
            output_path = Path(clip_results_dir) / f'{Path(current_video_path).stem}.mp4'
            processed_segment_paths.append(str(output_path))
            
            if isinstance(audio, tuple):
                # If returning two audio streams (main audio and background audio)
                main_audio, bg_audio = audio
                make_video(video_info, output_path, main_audio, sampling_rate=seq_cfg.sampling_rate)
                
                temp_bg_audio = tempfile.NamedTemporaryFile(suffix='.flac', delete=False).name
                temp_main_audio = tempfile.NamedTemporaryFile(suffix='.flac', delete=False).name

                torchaudio.save(temp_bg_audio, bg_audio, seq_cfg.sampling_rate)
                torchaudio.save(temp_main_audio, main_audio, seq_cfg.sampling_rate)

                bg_segment = AudioSegment.from_file(temp_main_audio)
                main_segment  = AudioSegment.from_file(temp_main_audio)
                    
                main_duration_ms = len(main_segment)
                bg_duration_ms = len(bg_segment)

                if bg_duration_ms > main_duration_ms:
                    # Trim background audio to match main audio length
                    bg_segment = bg_segment[:main_duration_ms]

                bg_segment = bg_segment
                
                # Overlay audio segments
                final_audio = main_segment.overlay(bg_segment)
                # final_audio = main_segment

                replace_video_audio(
                    video_path=output_path,
                    audio_segment=final_audio,
                    output_path=str(output_path),
                    sampling_rate=seq_cfg.sampling_rate
                )

                os.unlink(temp_bg_audio)
                os.unlink(temp_main_audio)

            else:
                final_audio = audio
                
                make_video(video_info, output_path, final_audio, sampling_rate=seq_cfg.sampling_rate)
        
        if video_save_path is None:
            video_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    merge_videos_in_directory(clip_results_dir, video_save_path)
    log.info(f'Saved video to {video_save_path}')
    
    if not return_segments:
        cleanup_temp_dir(seg_dir)
        cleanup_temp_dir(keyframes_dir)
        cleanup_temp_dir(clip_results_dir)
        return video_save_path
    else:
        # Return both final video and segment paths
        return video_save_path, processed_segment_paths

import shutil
def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory and its contents"""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Successfully removed temp directory: {temp_dir}")
    except Exception as e:
        logging.error(f"Error removing temp directory {temp_dir}: {e}")

@torch.inference_mode()
def text_to_audio(prompt: str, negative_prompt: str, seed: int, num_steps: int, cfg_strength: float,
                  duration: float):
    
    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    
    audio_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.flac').name
    torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
    log.info(f'Saved audio to {audio_save_path}')
    
    return audio_save_path

def video2audio_generation(video, prompt, negative_prompt, duration, num_steps, cfg_strength, rng, clip_results_dir=None):

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    log.info(f'Saved video to {video}')
    
    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames, sync_frames, [prompt], negative_text=[negative_prompt],
                      feature_utils=feature_utils, net=net, fm=fm, rng=rng, cfg_strength=cfg_strength)
    
    audio = audios.float().cpu()[0]

    if clip_results_dir is not None:
        video = Path(video)
        video_save_path = clip_results_dir / f'{video.stem}.mp4'

        if not os.path.exists(clip_results_dir):
            print('Save path does not exist')
        else:
            print('Save path exists')

        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
    
    return audio

@torch.inference_mode()
def text2audio_generation(prompt, negative_prompt, duration, num_steps, cfg_strength, rng, clip_results_dir=None):
    
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    
    audios = generate(clip_frames, sync_frames, [prompt], negative_text=[negative_prompt],
                      feature_utils=feature_utils, net=net, fm=fm, rng=rng, cfg_strength=cfg_strength)
    
    audio = audios.float().cpu()[0]
    if clip_results_dir is not None:
        audio_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.flac').name
        torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
        log.info(f'Saved audio to {audio_save_path}')
    
    return audio

@torch.inference_mode()
def layer_audio_gen(
    video_path: str,
    audio_description: Dict[str, List[str]],
    duration: float,
    negative_prompt: str = "",
    seed: int = -1,
    num_steps: int = 25,
    cfg_strength: float = 4.5,
    **kwargs
) -> torch.Tensor:

    background = audio_description.get("background audio", [])
    main_audio = audio_description.get("audio", [])
    
    background_prompt = ", ".join(background) if background else ""
    main_audio_prompt = ", ".join(main_audio) if main_audio else ""
    
    is_background_empty = not background or background == ["None"] or background == ['None'] or background == ['']
    is_audio_empty = not main_audio or main_audio == ["None"] or main_audio == ['None'] or main_audio == ['']
    
    rng = torch.Generator(device=device)
    if seed >= 0:
        rng.manual_seed(seed)
    else:
        rng.seed()
    
    try:
        if is_background_empty and not is_audio_empty:
            # Generate main audio only
            print("Generating main audio only...")
            final_audio = video2audio_generation(
                video=video_path, prompt=main_audio_prompt, negative_prompt=negative_prompt, duration=duration, 
                num_steps=num_steps, cfg_strength=cfg_strength, rng=rng, clip_results_dir=None  # Don't save video
            )
        
        elif not is_background_empty and is_audio_empty:
            # Generate background audio only
            print("Generating background audio only...")
            final_audio = video2audio_generation(
                video=video_path, prompt=background_prompt, negative_prompt=negative_prompt, duration=duration,  
                num_steps=num_steps, cfg_strength=cfg_strength, rng=rng, clip_results_dir=None
            )
        
        elif not is_background_empty and not is_audio_empty:
            # Generate two types of audio
            print("Generating both background and main audio...")
            background_audio = text2audio_generation(
                prompt=background_prompt, negative_prompt=negative_prompt, duration=duration, num_steps=num_steps, 
                cfg_strength=cfg_strength, rng=rng, clip_results_dir=None
            )
            
            main_audio = video2audio_generation(
                video=video_path, prompt=main_audio_prompt, negative_prompt=negative_prompt, duration=duration, 
                num_steps=num_steps, cfg_strength=cfg_strength, rng=rng, clip_results_dir=None
            )
            
            # Return two audio streams
            final_audio = (main_audio, background_audio)

        elif is_background_empty and is_audio_empty:
            final_audio = video2audio_generation(
                video=video_path, prompt="", negative_prompt=negative_prompt, duration=duration, 
                num_steps=num_steps, cfg_strength=cfg_strength, rng=rng, clip_results_dir=None
            )
            
        else:
            print("No valid audio description provided")
            return None
            
        return final_audio
        
    except Exception as e:
        print(f"Error generating layered audio: {str(e)}")
        return None

video_to_audio_tab = gr.Interface(
    fn=video_to_audio,
    inputs=[
        gr.Video(),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt', value='music'),
        gr.Number(label='Seed (-1: random)', value=-1, precision=0, minimum=-1),
        gr.Number(label='Num steps', value=25, precision=0, minimum=1),
        gr.Number(label='Guidance Strength', value=4.5, minimum=1),
        gr.Number(label='Duration (sec)', value=8, minimum=1),
    ],
    outputs='playable_video',
    cache_examples=False,
    title='Auto Video Audio — Long Video-to-Audio Synthesis',
    examples=[
        [
            './demo/001.mp4',
            '',
            '',
            0,
            25,
            4.5,
            23,
        ],
    ]
)

def create_segment_player_interface():
    with gr.Blocks(title="Video Segment Player") as segment_player:
        gr.Markdown("Video")
        
        video_segments_paths_state = gr.State([])
        video_captions_state = gr.State({})
        current_index_state = gr.State(0)
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video")
                process_btn = gr.Button("Process Video")
            
            with gr.Column(scale=2):
                
                with gr.Group():
                    segment_video = gr.Video(label="Video Segment", height=400)
                    segment_caption = gr.TextArea(label="Segment Caption", lines=4, interactive=False)
                    
                    with gr.Row():
                        prev_btn = gr.Button("Previous Segment")
                        segment_counter = gr.Markdown("Segment 0/0")
                        next_btn = gr.Button("Next Segment")
        
        def process_segments(video):
            if not video:
                return [], {}, None, "", "Segment 0/0", 0
            
            try:
                # Call video_process to process the video
                video_segments_paths, video_captions = video_process(video)
                
                if not video_segments_paths:
                    return [], {}, None, "No Segment", "Segment 0/0", 0
                
                # Get the first video segment
                first_video = video_segments_paths[0]
                first_caption = video_captions.get(first_video, "No description")
                
                return (
                    video_segments_paths, 
                    video_captions, 
                    first_video, 
                    first_caption,
                    f" Segment 1/{len(video_segments_paths)}",
                    0  # Current index
                )
            except Exception as e:
                logging.error(f"Error processing video: {e}")
                return [], {}, None, f"Processing error: {str(e)}", "Segment 0/0", 0
        

        def prev_segment(video_segments_paths, video_captions, current_index):
            if not video_segments_paths:
                return None, "No previous segment", "Segment 0/0", current_index
            
            prev_index = (current_index - 1) % len(video_segments_paths)
            video_path = video_segments_paths[prev_index]
            caption = video_captions.get(video_path, "No description")
            
            return (
                video_path,
                caption,
                f" Segment {prev_index + 1}/{len(video_segments_paths)}",
                prev_index
            )

        def next_segment(video_segments_paths, video_captions, current_index):
            if not video_segments_paths:
                return None, "No video segment", "Segment 0/0", current_index
            
            next_index = (current_index + 1) % len(video_segments_paths)
            video_path = video_segments_paths[next_index]
            caption = video_captions.get(video_path, "No description")
            
            return (
                video_path,
                caption,
                f"Segment {next_index + 1}/{len(video_segments_paths)}",
                next_index
            )

        process_btn.click(
            process_segments,
            inputs=[video_input],
            outputs=[
                video_segments_paths_state, 
                video_captions_state,
                segment_video,
                segment_caption,
                segment_counter,
                current_index_state
            ]
        )
        
        prev_btn.click(
            prev_segment,
            inputs=[
                video_segments_paths_state,
                video_captions_state,
                current_index_state
            ],
            outputs=[
                segment_video,
                segment_caption,
                segment_counter,
                current_index_state
            ]
        )
        
        next_btn.click(
            next_segment,
            inputs=[
                video_segments_paths_state,
                video_captions_state,
                current_index_state
            ],
            outputs=[
                segment_video,
                segment_caption,
                segment_counter,
                current_index_state
            ]
        )
        
    return segment_player

if __name__ == "__main__":
    gr.TabbedInterface(
        [video_to_audio_tab, create_segment_player_interface()],
        ['Video-to-Audio', 'Video Segment Player']
    ).launch(allowed_paths=[output_dir], 
             show_error=True, 
             server_name="0.0.0.0", 
             server_port=9997)
