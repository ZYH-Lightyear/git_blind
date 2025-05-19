import os
from openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from utils.tools import load_prompt
# from utils.tools import load_prompt
from typing import Optional, Dict, Any, List, Union
import json
from .base import BaseLLM
from utils.label_rag.llamaindex_rag import LabelRAG

class AudioPromptOptimizer(BaseLLM):
    def __init__(
        self,
        model_name: str = "qwen-max",
        api_base: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None,
        llm_prompt_path: str = "./prompt/audio_prompt_optimize.txt",
        aduio_lllm_rag_prompt_path: str = "./prompt/audio_llm_rag.txt"
    ) -> None:
        super().__init__(model_name)
        self.api_base = api_base
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.client = None

        self.system_prompt = load_prompt(llm_prompt_path)
        self.init_model()

        # Define the LLM for driving retrieval
        self.llama_index_llm = OpenAILike(
            model=self.model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            is_chat_model=True,
            system_prompt=load_prompt(aduio_lllm_rag_prompt_path)
        )

        self.label_rag = LabelRAG(storage_dir="./audio_label_documents/rag_index/", llm=self.llama_index_llm)

        # Initialize or load index
        if not self.label_rag.load_index():
            self.label_rag.create_index("./audio_label_documents/vggsound_labels.txt")
            self.label_rag.save_index()
    
    def init_model(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def get_better_audio_label(self, video_caption ,audio_description: str) -> str:
        print("call get_better_audio_label function")
        response = self.label_rag.query(video_caption, audio_description)
        return f"{response}"
    
    def get_completion(self, messages: List[Dict[str, str]],format="text", load_tools=True):
        """Get conversation completion"""
        kwargs = {"model": self.model_name, "messages": messages, "response_format": {"type": format}}
        
        completion = self.client.chat.completions.create(**kwargs)
        return completion.model_dump()
    
    
    def generate(self, video_caption: str, audio_description: str) -> str:
        # RAG retrieval for suitable labels
        tool_response = self.get_better_audio_label(video_caption, audio_description)
        print(tool_response)
        
        self.system_prompt = load_prompt("./prompt/audio_prompt_optimize.txt")
        
        # Initial conversation
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f'''
                ** Input **:
                - Audio Label Alternative Reference: {tool_response}
                - Video Caption: {video_caption}
                - Audio Description: {audio_description}
                ** Note **:
                The output audio description tag can have at most one
                '''
            },
        ]
        
        first_response = self.get_completion(messages, format="json_object")
        assistant_output = first_response['choices'][0]['message']
        print(assistant_output['content'])
        return assistant_output['content']
