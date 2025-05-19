from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.tools import load_prompt
# from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import load_index_from_storage
from typing import Dict, Any

def completion_to_prompt(completion):
   return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt

class LabelRAG:
    def __init__(self, storage_dir="./storage/VGGSound/", llm = None):
        self.storage_dir = storage_dir
        self.llm = llm
        self._setup_settings()
        self.index = None
    
    def _setup_settings(self):
        """初始化LLM和Embedding设置"""
        Settings.llm = self.llm
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5"
        )
        Settings.transformations = [SentenceSplitter(chunk_size=1024)]
    
    def create_index(self, document_path):
        """从文档创建新索引"""
        docs = SimpleDirectoryReader(input_files=[document_path]).load_data()
        self.index = VectorStoreIndex.from_documents(
            docs,
            show_progress=True,
            embed_model=Settings.embed_model,
            transformations=Settings.transformations
        )
        return self.index
    
    def save_index(self):
        """保存索引到本地"""
        if self.index:
            self.index.storage_context.persist(persist_dir=self.storage_dir)
    
    def load_index(self):
        """从本地加载索引"""
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.storage_dir
            )
            self.index = load_index_from_storage(storage_context)
            return True
        except Exception as e:
            print(f"加载索引失败: {str(e)}")
            return False
    
    def _format_query(self, input_data: Dict[str, Any]) -> str:
            """格式化查询字符串"""
            video_caption = input_data.get("video_caption", "")
            audio_desc = input_data.get("audio_description", "")

            query_note_prompt = load_prompt("./prompt/audio_llm_rag.txt")
            
            return f"""
            ** Input **:
            - Video Caption: {video_caption}
            - Audio Description: {audio_desc}

            ** Note **:
            {query_note_prompt}
            """

    def query(self, video_caption: str, audio_description: str):
        """查询索引并使用Agent处理响应
        Args:
            video_caption: 视频描述
            audio_description: JSON形式的音频描述字符串
        """
        if not self.index:
            raise ValueError("The index is not initialized, please create or load the index first")
        
        input_data = {
            "video_caption": video_caption,
            "audio_description": audio_description
        }
        query_engine = self.index.as_query_engine()
        query = self._format_query(input_data)
        response = query_engine.query(query).response
        
        return response