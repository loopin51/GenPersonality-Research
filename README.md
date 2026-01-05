# Child Personality Development Simulation (LLM & RL)

이 프로젝트는 거대 언어 모델(LLM)과 강화학습(RL)을 결합하여, **양육 환경(다정함 vs 차가움)이 유아의 성격 형성에 미치는 영향**을 컴퓨터 시뮬레이션으로 모델링한 연구용 애플리케이션입니다.

## 📌 주요 특징 (Key Features)

* **인지적 에이전트 (Cognitive Child Agent)**:
  * **감정 (Emotion)**: PAD (Pleasure, Arousal, Dominance) 3차원 벡터로 기분 상태를 실시간 모델링.
  * **기억 (Memory)**: ChromaDB를 활용한 일화 기억(Episodic Memory) 저장 및 가중 검색 (유사도 + 감정 강도).
  * **핵심 신념 (Core Beliefs)**: 주기적인 자아 성찰(Reflection)을 통해 경험을 일반화된 신념으로 형성.
  * **행동 결정 (Decision Making)**: DQN(Deep Q-Network)을 통해 감정 상태에 따른 최적의 행동 전략(10가지 유형) 학습.
* **부모 페르소나 (Parent Persona)**: LLM 프롬프트 엔지니어링으로 일관된 양육 태도(Warm/Cold) 구현.
* **장기 시뮬레이션**: 3단계 발달 과정(탐색기-습관형성기-고착기)을 거치는 150 에피소드 시뮬레이션.
* **평가 시스템**: 시뮬레이션 종료 후 **낯선 상황 실험(Strange Situation Test)**을 자동으로 수행하여 애착 유형 진단.

## 📂 디렉토리 구조

```
ChildLLM/
├── data/
│   ├── scenarios.json       # 기본 시나리오 뱅크
│   └── episodes.json        # 생성된 150개 에피소드 플레이리스트
├── src/
│   ├── agents/              # Child, Parent 에이전트 로직
│   ├── models.py            # 데이터 구조 (PAD, ActionType 등)
│   ├── memory.py            # ChromaDB 메모리 시스템
│   ├── simulation.py        # 시뮬레이션 엔진 및 루프
│   ├── evaluation.py        # 낯선 상황 테스트 평가 모듈
│   └── config.py            # 환경 설정 및 상수
├── log/                     # 시뮬레이션 결과 로그 (CSV) - 자동 생성
├── output/                  # 학습된 모델(.pth) 및 진단 리포트(.txt) - 자동 생성
├── main.py                  # 메인 실행 파일
└── requirements.txt         # 의존성 패키지
```

## 🚀 설치 및 설정 (Installation & Setup)

1. **필수 패키지 설치**

    ```bash
    pip install -r requirements.txt
    ```

2. **환경 변수 설정 (`.env`)**
    프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 작성하세요. (OpenRouter 또는 OpenAI 사용 가능)

    ```bash
    # OpenRouter 사용 시 (권장)
    OPENROUTER_API_KEY=sk-or-your-key-here
    MODEL_NAME=openai/gpt-3.5-turbo  # 또는 anthropic/claude-3-haiku 등
    YOUR_SITE_URL=https://your-site.com (선택)
    YOUR_SITE_NAME=ChildLLM (선택)
    ```

## ▶️ 실행 방법 (Usage)

시뮬레이션을 시작하려면 메인 스크립트를 실행하세요.

```bash
python main.py
```

실행 후 프롬프트에서 부모의 양육 타입을 선택합니다:

* `1`: **Warm (다정하고 수용적인 부모)**
* `2`: **Cold (차갑고 엄격한 부모)**

## 📊 결과 확인 (Outputs)

시뮬레이션(150 에피소드)이 완료되면 다음 파일들이 생성됩니다.

1. **로그 파일 (`log/`)**:
    * `sim_results_{RunType}_{Timestamp}.csv`: 매 에피소드별 감정 상태, 리워드, 행동, 사용된 엡실론 값 등이 기록된 원시 데이터.
2. **학습 모델 (`output/`)**:
    * `child_model_{RunType}_{Timestamp}.pth`: 학습이 완료된 아이 에이전트의 신경망(DQN) 가중치 파일. (성격 그 자체)
3. **진단 리포트 (`output/`)**:
    * `diagnosis_{RunType}_{Timestamp}.txt`: **낯선 상황 실험**을 통해 진단된 최종 애착 유형(안정/회피/저항) 및 행동 반응 요약.

## 🧠 실험 과정 요약

1. **Phase 1 (Ep 1~30)**: 무작위 탐색. 아이는 다양한 행동을 시도하며 부모의 반응을 살핍니다.
2. **Phase 2 (Ep 31~100)**: 패턴 형성. 부모의 반응에 맞춰 특정 행동 빈도가 늘어납니다. 10회마다 **Reflection**이 발생하여 "신념(Belief)"이 형성됩니다.
3. **Phase 3 (Ep 101~150)**: 성격 고착. 학습된 행동 패턴이 굳어지며 새로운 시도(Epsilon)가 최소화됩니다.

---
**Note**: 이 프로젝트는 심리학적 이론(Karen Horney, Mary Ainsworth 등)을 기반으로 설계되었으나, 실제 아동 심리를 완벽하게 대변하지는 않습니다. 연구 및 실험 목적으로만 사용하세요.
