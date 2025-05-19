from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.tools import load_prompt
from typing import List, Optional
from qwen_vl_utils import process_vision_info
import torch
from typing import List, Optional, Tuple, Any
from .base import VideoUnderstandBase

class Qwen2_5VideoUnderstand(VideoUnderstandBase):
    def init_model(self) -> Tuple[Any, Any]:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            min_pixels=256*256, 
            max_pixels=512*512,
            revision="refs/pr/24"
        )
        self.model = model
        self.processor = processor
        return model, processor

    def inference(self, frames: List[str], prompt: Optional[str] = None, whole_video_caption = None) -> str:
        try:
            if self.model is None or self.processor is None:
                self.init_model()
            if whole_video_caption is not None:
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "text", 
                            "text": "whole_video_caption: \n" + whole_video_caption
                        },
                    ],
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                        },
                        {
                            "type": "text", 
                            "text": prompt or "这段视频里发生了什么?"
                        },
                    ],
                }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda:0")
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            return f"Error during video inference: {str(e)}"
        
    def inference_whole_video(self, video_path, prompt: Optional[str] = None) -> str:
        try:
            if self.model is None or self.processor is None:
                self.init_model()
            # file:///path/to/video1.mp4
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": 'file://' + video_path,
                        "max_pixels": 480 * 480,
                        "fps": 1.0,
                    },
                    {
                        "type": "text", 
                        "text": load_prompt("./prompt/whole_video_understand.txt")
                    },
                ],
            }]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            return f"Error during video inference: {str(e)}"

# def init_qwen_vl_model():
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
#     )

#     min_pixels = 256*28*28
#     max_pixels = 512*28*28
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

#     return model, processor

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.

# def qwen_video_inference(
#     images_path: str, 
#     model: Optional[Qwen2_5_VLForConditionalGeneration] = None,
#     processor: Optional[AutoProcessor] = None,
#     prompt = None
# ) -> str:
#     try:
#         # Initialize model if not provided
#         if model is None or processor is None:
#             model, processor = init_qwen_vl_model()

#         # user_prompt = load_prompt("./prompt/video_understand.txt")

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": [],
#                         "fps": 1.0,
#                     },
#                     {"type": "text", "text": "这个视频发生了什么"},
#                 ],
#             }
#         ]

#         # Preparation for inference
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to(model.device)

#         # Inference: Generation of the output
#         generated_ids = model.generate(**inputs, max_new_tokens=128)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         # print(output_text)
        
#         return output_text[0]
        
#     except Exception as e:
#         return f"Error during video inference: {str(e)}"
    
# def main():
#     path = '/hpc2hdd/home/yzhang679/codes/vid_audio/test_seg/aigc_1/aigc_1-Scene-001.mp4'
#     video_path = 'file://' + path
#     model, processor = init_qwen_vl_model()
#     result = qwen_video_inference(video_path, model, processor)
    
#     print(result)
    
# if __name__ == "__main__":
#     main()



"""
    Input: Provide a video and ask the system to analyze its scene, approximate timeline, and key events.

    Objective:
    Key Event Extraction: Extracting important content from videos, especially those that generate sound, such as explosions.
    Person and object detection: Detect the main characters, objects, and their interactions in a video.
    Video summary: Generate a brief text that summarizes the video content and extracts the core information.

    Output:
    "Description of no more than 100 words"
"""