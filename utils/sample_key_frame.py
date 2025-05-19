import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from typing import List, Dict

class KeyFrameExtractor:
    def __init__(self, video_path: str, base_dir: str = "./results"):
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.frames = []
        self.shot_boundaries = []
        self.keyframes = []

    def preprocess_video(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        return self

    def detect_shot_boundaries(self, threshold=10):
        for i in range(1, len(self.frames)):
            prev_frame = cv2.cvtColor(self.frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            diff = np.mean(np.abs(curr_frame.astype(int) - prev_frame.astype(int)))
            if diff > threshold:
                self.shot_boundaries.append(i)
        return self

    def cal_frame_nums(self) -> int:
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        if duration <= 1.5:
            return 4
        else:
            return 4 + int((duration - 1.5) * 1.5)
    
    def _sample_frames(self, frames, target_fps=8):
        cap = cv2.VideoCapture(self.video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if original_fps <= target_fps:
            return frames, list(range(len(frames)))

        step = int(original_fps / target_fps)
        sampled_frames = []
        sampled_indices = []
        
        for i in range(0, len(frames), step):
            sampled_frames.append(frames[i])
            sampled_indices.append(i)
        
        return sampled_frames, sampled_indices
    
    def extract_keyframes(self):
        shot_frames, original_indices = self._sample_frames(self.frames[2:], target_fps=8)
        n_clusters = self.cal_frame_nums()
        
        frame_features = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten() 
                                 for frame in shot_frames])

        n_clusters = min(n_clusters, len(frame_features))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(frame_features)

        for cluster_idx in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            distances = np.sum((frame_features - cluster_center) ** 2, axis=1)
            frame_idx = np.argmin(distances)
            
            self.keyframes.append({
                'frame': shot_frames[frame_idx],
                'frame_idx': original_indices[frame_idx],
            })
        
        self.keyframes.sort(key=lambda x: x['frame_idx'])
        return self
    
    def extract_average_keyframes(self):
        if not self.frames:
            self.preprocess_video()
        
        n_frames = self.cal_frame_nums()
        n_frames = min(n_frames, len(self.frames))
        
        if n_frames <= 1:
            middle_idx = len(self.frames) // 2
            self.keyframes = [{
                'frame': self.frames[middle_idx],
                'frame_idx': middle_idx
            }]
            return self
        interval = len(self.frames) / n_frames
        
        self.keyframes = []
        for i in range(n_frames):
            frame_idx = min(int(i * interval), len(self.frames) - 1)
            self.keyframes.append({
                'frame': self.frames[frame_idx],
                'frame_idx': frame_idx
            })
        
        return self
    
    def save_keyframes(self):

        metadata = []
        for i, kf in enumerate(self.keyframes):

            filename = f'keyframe_{i:04d}_frame_{kf["frame_idx"]}.jpg'
            filepath = os.path.join(self.output_dir, filename)

            cv2.imwrite(filepath, kf['frame'])

            metadata.append({
                'keyframe_id': i,
                'frame_index': kf['frame_idx'],
                'filename': filename
            })
        
        print(f"Saved {len(self.keyframes)} keyframes to {self.output_dir}")
        return metadata

    def kmeans_process(self):
        return (self.preprocess_video()
                .extract_keyframes()
                .save_keyframes())
    
    def average_process(self):
        return (self.preprocess_video()
                .extract_average_keyframes()
                .save_keyframes())

def process_single_video(video_path: str, output_base_dir: str = "./results") -> List[Dict]:
    try:
        print(f"\nProcessing video: {os.path.basename(video_path)}")
        extractor = KeyFrameExtractor(video_path, output_base_dir)
        metadata = extractor.kmeans_process()
        print(f"Video processing completed, extracted {len(metadata)} keyframes")
        return metadata
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return []

def process_with_average_sampling(video_path: str, output_base_dir: str = "./results") -> List[Dict]:
    try:
        print(f"\nProcessing video (uniform sampling): {os.path.basename(video_path)}")
        extractor = KeyFrameExtractor(video_path, output_base_dir)
        metadata = extractor.average_process()
        print(f"Video processing completed, extracted {len(metadata)} keyframes")
        return metadata
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return []

def process_video_folder(input_folder: str, output_base_dir: str = "./results") -> Dict[str, List]:
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_folder) 
                  if os.path.splitext(f)[1].lower() in video_extensions]
    video_files.sort()
    
    results = {}
    total_videos = len(video_files)
    print(f"\nFound {total_videos} video files")
    
    os.makedirs(output_base_dir, exist_ok=True)

    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(input_folder, video_file)
        print(f"\nProcessing video [{i}/{total_videos}]: {video_file}")
        try:
            extractor = KeyFrameExtractor(video_path, output_base_dir)
            metadata = extractor.average_process()
            results[video_file] = metadata
            
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")
            results[video_file] = []
            continue
    
    print("\nProcessing completed:")
    for video_file, metadata in results.items():
        print(f"- {video_file}: extracted {len(metadata)} keyframes")
    
    return results

def process_video_folder_with_average_sampling(input_folder: str, output_base_dir: str = "./results") -> Dict[str, List]:
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(input_folder) 
                  if os.path.splitext(f)[1].lower() in video_extensions]
    video_files.sort()
    
    results = {}
    total_videos = len(video_files)
    print(f"\nFound {total_videos} video files")
    
    os.makedirs(output_base_dir, exist_ok=True)

    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(input_folder, video_file)
        print(f"\nProcessing video [{i}/{total_videos}]: {video_file} (uniform sampling)")
        try:
            extractor = KeyFrameExtractor(video_path, output_base_dir)
            metadata = extractor.average_process()
            results[video_file] = metadata
            
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")
            results[video_file] = []
            continue
    
    print("\nProcessing completed:")
    for video_file, metadata in results.items():
        print(f"- {video_file}: extracted {len(metadata)} keyframes")
    
    return results