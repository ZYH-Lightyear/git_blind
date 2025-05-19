from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple

class VideoUnderstandBase(ABC):
    def __init__(self):
        self.model = None
        self.processor = None
        
    @abstractmethod
    def init_model(self) -> Tuple[Any, Any]:
        """Initialize model and processor"""
        pass
    
    @abstractmethod
    def inference(self, frames: List[str], prompt: Optional[str] = None) -> str:
        """Run inference on video frames
        
        Args:
            frames: List of frame paths
            prompt: Optional prompt template
        Returns:
            str: Generated description
        """
        pass
