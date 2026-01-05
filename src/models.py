from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from enum import Enum

@dataclass
class PADState:
    """
    Represents the internal emotional state of the child agent.
    Range: -1.0 ~ +1.0 for each dimension.
    """
    P: float = 0.0  # Pleasure-Displeasure
    A: float = 0.0  # Arousal-Nonarousal
    D: float = 0.0  # Dominance-Submissiveness

    def to_vector(self) -> np.ndarray:
        return np.array([self.P, self.A, self.D], dtype=np.float32)
    
    def clip(self):
        self.P = max(-1.0, min(1.0, self.P))
        self.A = max(-1.0, min(1.0, self.A))
        self.D = max(-1.0, min(1.0, self.D))
        
    def decay(self, rate: float):
        """Applies decay towards baseline (0.0). Formula: E(t+1) = E(t) * (1 - lambda)"""
        self.P *= (1.0 - rate)
        self.A *= (1.0 - rate)
        self.D *= (1.0 - rate)

class ActionType(Enum):
    """
    10 Discrete Actions based on Karen Horney's theory.
    Mapped to index 0-9.
    """
    COMPLIANCE_ACTIVE = 0  # 적극적 순응
    AFFECTION_SEEKING = 1  # 애정 갈구
    APOLOGY_PLEA = 2       # 사과/호소
    REFUSAL_TANTRUM = 3    # 거부/떼쓰기
    AGGRESSION_BLAME = 4   # 비난/공격
    CONTROL_ATTEMPT = 5    # 통제 시도
    SILENCE_IGNORE = 6     # 묵묵부답/무시
    PHYSICAL_AVOIDANCE = 7 # 물리적 회피
    SADNESS_EXPRESSION = 8 # 슬픔 표현
    EXPLANATION = 9        # 상황 설명

    @classmethod
    def get_name(cls, index: int) -> str:
        return cls(index).name

    @classmethod
    def get_description(cls, index: int) -> str:
        descriptions = {
            0: "지시를 즉시 따르고 긍정적으로 대답함",
            1: "사랑해달라고 하거나 스킨십/칭찬 요구",
            2: "잘못했다고 빌거나 두려움을 표현하며 애원",
            3: "\"싫어!\"라고 소리치거나 떼를 씀",
            4: "\"엄마 때문이야\"라며 남 탓을 하거나 공격",
            5: "\"이거 해주면 할게\"라며 조건/협상 제시",
            6: "대답을 안 하거나 못 들은 척함",
            7: "방에 들어가거나 자리를 피하겠다고 함",
            8: "서럽게 울거나 슬픈 감정을 솔직히 말함",
            9: "왜 그랬는지 이유를 차분히 설명함"
        }
        return descriptions.get(index, "Unknown Action")

@dataclass
class MemoryItem:
    """
    Episodic Memory Schema as defined in Project_receipt.md 3.1
    """
    episode_id: str
    trigger: str
    action: str  # Action Name/Description
    outcome: str # Parent Response
    emotion_impact: List[float] # [Delta P, Delta A, Delta D]
    timestamp: int # or episode number
    embedding: Optional[List[float]] = None

@dataclass
class SimulationLog:
    """For tracking history"""
    episode_num: int
    scenario_id: str
    parent_type: str
    initial_pad: PADState
    final_pad: PADState
    action_idx: int
    reward: float
    transcript: List[Dict[str, str]] # [{'role': 'parent', 'text': ...}, ...]
