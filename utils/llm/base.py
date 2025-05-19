from abc import ABC, abstractmethod
import json
from typing import Optional, Dict, Any

class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        
    @abstractmethod
    def init_model(self):
        """Initialize the model"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        pass
        
    def parse_json(self, text: str) -> Dict[str, Any]:
        """Helper method to parse JSON from response"""
        try:
            # Find JSON content between curly braces
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return {}
