import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.tools import load_prompt
from typing import Optional, Dict, Any, List
import json
from .base import BaseLLM

load_dotenv()

# InitialAudioPromptGen通过CoT Prompt，接收视频片段的文字描述，生成初步的Audio Prompt
class InitialAudioPromptGen(BaseLLM):
    """A wrapper class to interact with a language model."""
    
    def __init__(
        self,
        model_name: str = "qwen-max",
        api_base: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None,
        prompt_path: str = "./prompt/designer_initial.txt"
    ) -> None:
        """
        Initialize the LLM.
        Args:
            model_name: 模型名称
            api_base: API基础URL
            api_key: API密钥，如果为None则从环境变量获取Qwen API密钥
            prompt_path: 系统提示词文件路径
        """
        super().__init__(model_name)
        self.api_base = api_base
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.init_model()
        self.system_prompt = load_prompt(prompt_path)
        self.prompt_path = prompt_path
        
    def init_model(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )

    def get_completion(self, messages: List[Dict[str, str]],format="text", load_tools=True):
        """获取对话补全"""
        kwargs = {"model": self.model_name, "messages": messages, "response_format": {"type": format}}
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.model_dump()
    
    def refine_audio_label_multi_round(
        self, 
        video_caption: str, 
        audio_description: str, 
        audio_generator=None,
        max_rounds: int = 3
    ) -> str:
        """
        Refine audio labels through multiple rounds of dialogue.
        
        Args:
            video_caption: The caption describing the video content
            audio_description: Initial audio description/label
            audio_generator: Instance of AudioPromptOptimizer to call get_better_audio_label
            max_rounds: Maximum number of refinement rounds (default: 3)
            
        Returns:
            str: The refined audio label after multiple rounds
        """
        if audio_generator is None:
            raise ValueError("Audio generator must be provided to use multi-round refinement")
        
        current_label = audio_description
        reference_label = audio_generator.get_better_audio_label(video_caption, audio_description)
        
        print(f"Starting multi-round label refinement process")
        print(f"Initial label: {current_label}")
        print(f"Reference label: {reference_label}")
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n--- Round {round_num}/{max_rounds} ---")
            
            # Prepare messages for refinement
            messages = [
                {
                    "role": "system",
                    "content": "You are an audio label optimization assistant. Your task is to refine and improve audio labels based on video context and reference labels. Consider specificity, accuracy, and relevance in your refinement."
                },
                {
                    "role": "user",
                    "content": f"""
                    Please refine this audio label:
                    
                    Video caption: {video_caption}
                    Current audio label: {current_label}
                    Reference label: {reference_label}
                    
                    Analyze the current label and reference label, then provide an improved version that better describes the audio content based on the video context.
                    Return only the improved label text without any explanations or additional information.
                    """
                }
            ]
            
            # Get refined label from LLM
            response = self.get_completion(messages, format="text")
            refined_label = response['choices'][0]['message']['content'].strip()
            
            # Get new reference using the refined label
            new_reference = audio_generator.get_better_audio_label(video_caption, refined_label)
            
            print(f"Current label: {current_label}")
            print(f"Refined label: {refined_label}")
            print(f"New reference: {new_reference}")
            
            # Check if significant improvement was made
            if refined_label == current_label:
                print("No further refinement needed, ending process early.")
                break
            
            current_label = refined_label
            reference_label = new_reference
            
        print(f"\nFinal refined label after {round_num} rounds: {current_label}")
        return current_label
    
    def generate(
        self, 
        last_video_caption: str,
        prompt: str,
        response_format: str = "json_object",
        use_multi_round: bool = False,
        audio_generator=None,
        audio_description: str = None
    ) -> str:
        if self.client is None:
            self.init_model()
        
        if use_multi_round and audio_generator is not None and audio_description is not None:
            refined_audio = self.refine_audio_label_multi_round(
                video_caption=prompt, 
                audio_description=audio_description,
                audio_generator=audio_generator
            )
            prompt_content = f'''
            Last Video Caption: {last_video_caption}
            Current Video Caption: {prompt}
            Refined Audio Description: {refined_audio}
            '''
        else:
            prompt_content = f'''
            Last Video Caption: {last_video_caption}
            Current Video Caption: {prompt}
            '''
            
        self.system_prompt = load_prompt(self.prompt_path)
        # 初始对话
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": prompt_content
            },
        ]
        
        try:
            first_response = self.get_completion(messages, format="json_object")
            assistant_output = first_response['choices'][0]['message']
            print(assistant_output['content'])
            return assistant_output['content']
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return ""
    
    def parse_json(self, response: str):
        """
        解析JSON格式的回复
        Args:
            response: 模型生成的回复
        Returns:
            Dict[str, Any]: 解析后的回复
        """
        return json.loads(response)



