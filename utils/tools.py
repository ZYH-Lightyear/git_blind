import ffmpeg
import os
import tempfile
import shutil
from typing import Optional
from datetime import datetime
from moviepy import *
import cv2
import numpy as np
from pathlib import Path
from pydub import AudioSegment, effects


# example 
# input_video = "/hpc2hdd/home/yzhang679/codes/vid_audio/results/aigc_1/aigc_1-Scene-133.mp4"
# convert_flac_to_mp3(input_video, input_video)

def merge_two_videos(video1_path: str, video2_path: str, output_path: str) -> None:
    """Merge two video clips"""
    try:
        video1 = VideoFileClip(video1_path)
        video2 = VideoFileClip(video2_path)
        final_clip = concatenate_videoclips([video1, video2])
        final_clip.write_videofile(output_path)
    except Exception as e:
        print(f"Error merging videos: {e}")
        raise e


def convert_flac_to_mp3(
    input_path: str, 
    output_path: Optional[str] = None
) -> bool:
    """
    Convert FLAC audio codec in MP4 to MP3
    Args:
        input_path: Path to input video file
        output_path: Path for output video file (optional)
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        if output_path is None:
            output_path = input_path
            
        # Create temporary file
        temp_output = os.path.join(
            tempfile.gettempdir(),
            f"temp_{os.path.basename(output_path)}"
        )
        
        # Convert using ffmpeg with temporary file
        (
            ffmpeg
            .input(input_path)
            .output(
                temp_output,
                acodec='mp3',
                vcodec='copy'
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Replace original file with converted file
        shutil.move(temp_output, output_path)
        
        print(f"Successfully converted {input_path}")
        return True
        
    except Exception as e:
        print(f"Error converting video: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False

def concat_video(video_paths, output_path):
    # Load video files
    video_clips = [VideoFileClip(video_path) for video_path in video_paths]

    # Concatenate videos on the timeline
    final_video = concatenate_videoclips(video_clips, method="compose")
    
    # Save the result
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print("Videos successfully concatenated!")

def load_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt

def get_sorted_files(directory):
    # Get absolute paths for all files in directory
    file_paths = [os.path.join(os.path.abspath(directory), file) 
                 for file in os.listdir(directory)]
    # Sort paths
    return sorted(file_paths)

def get_extended_filename(original_path, out_dir=None):
    """Generate extended filename while preserving original name"""
    original_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(original_name)[0]
    ext = os.path.splitext(original_name)[1]
    if out_dir is None:
        out_dir = os.path.dirname(original_path)
    return os.path.join(out_dir, f"{name_without_ext}_extended{ext}")

def concatenate_video(input_video_path, temp_output_path, target_duration=1.0):
    """Extend video by concatenating it with itself until reaching target duration"""
    try:
        video = VideoFileClip(input_video_path)
        duration = video.duration
        repeats = int(target_duration / duration) + 1
        
        # Create a temporary file for the concat operation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            # Write the file list for ffmpeg concat
            for _ in range(repeats):
                f.write(f"file '{input_video_path}'\n")
            f.flush()
            
            # Use ffmpeg concat demuxer
            stream = (
                ffmpeg
                .input(f.name, f='concat', safe=0)
                .output(temp_output_path, c='copy')
                .overwrite_output()
            )
            stream.run(quiet=True)
        
        video.close()
        return temp_output_path
        
    except Exception as e:
        print(f"Error in concatenate_video: {str(e)}")
        return None

def extend_short_video(input_video_path, temp_output_path):
    """Extend video duration to 1 second using multiple methods"""
    try:
        # Read video info using moviepy instead of ffprobe
        video = VideoFileClip(input_video_path)
        fps = video.fps
        duration = video.duration
        
        # Calculate the required speed factor to extend to 1 second
        speed_factor = duration  # This will slow down the video to reach 1 second
        
        # Use ffmpeg directly with setpts filter (slower playback)
        stream = (
            ffmpeg
            .input(input_video_path)
            .filter('setpts', f'{1/speed_factor}*PTS')  # Slow down the video
            .output(temp_output_path)
            .overwrite_output()
        )
        
        try:
            stream.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print("ffmpeg error occurred:", e.stderr.decode())
            # Fallback method: just copy the video multiple times
            with open(temp_output_path, 'wb') as outfile:
                for _ in range(int(1/duration) + 1):
                    with open(input_video_path, 'rb') as infile:
                        outfile.write(infile.read())
        
        video.close()
        return temp_output_path
        
    except Exception as e:
        print(f"Slow motion method failed: {str(e)}")
        print("Trying concatenation method...")
        
        # Try concatenation method
        result = concatenate_video(input_video_path, temp_output_path)
        if result:
            return result
            
        # If all methods fail, fall back to simple copy
        print("All extension methods failed, falling back to simple copy...")
        import shutil
        shutil.copy2(input_video_path, temp_output_path)
        return temp_output_path

def extend_short_video_by_frames(input_path: str, output_path: str = None) -> str:
    """Extend short video to 1 second by duplicating frames
    
    Args:
        input_path (str): Input video path
        output_path (str): Output video path, if None it will be generated automatically
        
    Returns:
        str: Output video path
    """
    if output_path is None:
        output_path = get_extended_filename(input_path)
        
    # Read video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate the total number of frames needed and the number of repetitions
    target_frames = int(fps)  # Number of frames in 1 second
    if total_frames >= target_frames:
        return input_path
        
    # Read all original frames
    original_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()
    
    # Calculate the number of times each original frame needs to be duplicated
    repeats = np.ceil(target_frames / total_frames)
    extended_frames = []
    
    # Duplicate frames
    for frame in original_frames:
        for _ in range(int(repeats)):
            extended_frames.append(frame)
    
    # If the number of frames exceeds the target, remove the extra frames
    extended_frames = extended_frames[:target_frames]
    
    # Write new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in extended_frames:
        out.write(frame)
    
    out.release()
    
    return output_path

def merge_videos_in_directory(input_dir, output_path):
    """
    Merge all mp4 videos in a directory in alphabetical order
    Args:
        input_dir: Directory containing mp4 files
        output_path: Path for the merged video file
    """
    try:
        # Get all mp4 files and sort them
        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        video_files.sort()  # Sort alphabetically
        
        if not video_files:
            print("No mp4 files found in directory")
            return None
            
        # Create temporary file for concat operation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            # Write the file list for ffmpeg concat
            for video in video_files:
                full_path = os.path.join(input_dir, video)
                f.write(f"file '{os.path.abspath(full_path)}'\n")
            f.flush()
            
            # Use ffmpeg concat demuxer
            try:
                stream = (
                    ffmpeg
                    .input(f.name, f='concat', safe=0)
                    .output(output_path, c='copy')
                    .overwrite_output()
                )
                stream.run(quiet=True)
                print(f"Successfully merged {len(video_files)} videos into {output_path}")
                return output_path
                
            except ffmpeg.Error as e:
                print(f"FFmpeg error during merge: {str(e)}")
                return None
                
    except Exception as e:
        print(f"Error in merge_videos: {str(e)}")
        return None

# Parse audio prompt (Dict format) output by LLM
def parse_audio_dict(audio_dict):
    result = []
    
    # Process background audio
    bg_audio = audio_dict.get("background audio", [])
    if bg_audio and bg_audio != ["None"] and bg_audio != "None":
        result.extend(bg_audio)
    
    # Process main audio
    main_audio = audio_dict.get("audio", [])
    if main_audio and main_audio != ["None"] and main_audio != "None":
        result.extend(main_audio)
    
    # Join all valid elements with commas
    return ', '.join(result) if result else ""


def replace_video_audio(video_path: str, audio_segment, output_path: str, sampling_rate: int = 44100):
    # Create temporary files with unique names
    temp_dir = Path(output_path).parent
    temp_audio = str(temp_dir / f"temp_audio_{datetime.now().timestamp()}.wav")
    temp_video = str(temp_dir / f"temp_video_{datetime.now().timestamp()}.mp4")
    
    try:
        # Export audio to temp file
        audio_segment.export(
            temp_audio, 
            format="wav",
            parameters=["-ar", str(sampling_rate)]
        )
        
        # Load video and audio
        video = VideoFileClip(video_path)
        audio = AudioFileClip(temp_audio)
        video.audio = audio
        
        # Write to temporary video file first
        video.write_videofile(
            temp_video,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            audio_fps=sampling_rate
        )
        
        # Replace original with new file
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_video, output_path)
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_audio, temp_video]:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")

def adjust_mp3_volume(input_mp3_path: str, output_mp3_path: str, target_dbfs: float) -> None:
    """
    Adjust the volume of MP3 audio to the target dBFS
    Args:
        input_mp3_path (str): Input MP3 file path
        output_mp3_path (str): Output MP3 file path
        target_dbfs (float): Target volume (unit: dBFS, usually negative, e.g., -20.0)
    """
    audio = AudioSegment.from_mp3(input_mp3_path)
    change_in_dBFS = target_dbfs - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_mp3_path, format="mp3")

def concat_audios(audio_paths, output_path):
    """
    Concatenate multiple audio files into one.
    Args:
        audio_paths (list): List of input audio file paths (must be the same format, e.g., all mp3 or all wav)
        output_path (str): Output audio file path (extension determines format)
    """
    if not audio_paths:
        raise ValueError("audio_paths list is empty.")
    # Load the first audio file
    combined = AudioSegment.from_file(audio_paths[0])
    # Append the rest
    for path in audio_paths[1:]:
        audio = AudioSegment.from_file(path)
        combined += audio
    # Export the result
    combined.export(output_path, format=output_path.split('.')[-1])

def denoise_audio(input_audio_path: str, output_audio_path: str, low_freq: int = 100, high_freq: int = 8000):
    """
    Simple audio denoising using high-pass and low-pass filters.
    Args:
        input_audio_path (str): Input audio file path (wav/mp3/etc.)
        output_audio_path (str): Output audio file path
        low_freq (int): High-pass filter cutoff frequency in Hz (default 100)
        high_freq (int): Low-pass filter cutoff frequency in Hz (default 8000)
    """
    audio = AudioSegment.from_file(input_audio_path)
    # Apply high-pass filter to remove low-frequency noise
    filtered = audio.high_pass_filter(low_freq)
    # Apply low-pass filter to remove high-frequency noise
    filtered = filtered.low_pass_filter(high_freq)
    # Optionally normalize volume
    filtered = effects.normalize(filtered)
    filtered.export(output_audio_path, format=output_audio_path.split('.')[-1])


