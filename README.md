# LLM Persona Generation Evaluation Framework

## 🎯 Project Overview
이 프로젝트는 **"회식 장소 선정 토론"**에 참여할 AI 에이전트의 **페르소나(Persona)를 생성하는 LLM 성능을 정량적으로 평가**하기 위한 프레임워크입니다.

단순히 "말을 잘하는지"를 보는 것이 아니라, **사용자의 복잡한 요구사항(알레르기, 호불호, 미묘한 뉘앙스)을 정확하게 파악하고, 이를 구조화된 데이터(JSON)로 변환할 수 있는지**를 검증합니다. 이를 통해 실제 서비스에 투입 가능한 최적의 모델(Accuracy vs Cost vs Speed)을 선정하는 것을 목표로 합니다.

---

## 📊 Evaluation Metrics Philosophy
우리는 모델 선정 시 다음과 같은 8가지 핵심 지표를 사용합니다. 각 지표는 서비스의 안정성과 품질에 직결되는 중요한 의미를 가집니다.

### 1. Stability & System Reliability (안정성)
서비스가 터지지 않고 굴러가기 위한 최소한의 조건입니다.

- **`json_schema_compliance` (JSON 구조 준수율)**
    - **Why?**: 아무리 똑똑한 답변도 JSON 형식이 깨지면 백엔드 파서(Parser)가 에러를 뱉고 서비스 장애로 이어집니다.
    - **Goal**: Syntax Error 없이 100% 완벽한 JSON을 생성하는지 검증합니다.

- **`field_coverage` (정보 누락 여부)**
    - **Why?**: 사용자가 입력한 필드(예: 30대, 남성)를 모델이 실수로 빠뜨리면, 엉뚱한 페르소나가 생성됩니다.
    - **Goal**: 입력된 모든 정보가 출력 결과에 하나도 빠짐없이 반영되었는지 확인합니다.

### 2. Logic & Reasoniong (논리적 판단력) 🧠
단순한 앵무새가 아니라, 맥락을 이해하는 지능을 평가합니다.

- **`classification_accuracy` (호불호 분류의 정확성)** ✨ *Critical*
    - **Why?**: **"알레르기(생명 직결)"**와 **"단순 불호(협상 가능)"**를 구분하는 것은 매우 중요합니다. 땅콩 알레르기 사용자를 "땅콩 싫어함(협상 가능)"으로 분류하면 큰일 납니다.
    - **Goal**: `must_avoid`(절대 금지)와 `preferred`(선호)를 정확히 구분해내는지 봅니다.

- **`consistency` (논리적 일관성)**
    - **Why?**: "돼지고기 알레르기 있음"이라고 해놓고 "삼겹살 선호"라고 출력하면 이는 **할루시네이션(환각)**입니다.
    - **Goal**: 입력 정보 간의 모순(알레르기 vs 선호 음식)을 스스로 감지하고 안전한 방향으로 해결하는지 평가합니다.

- **`extra_text_parsing` (비정형 텍스트 분석력)**
    - **Why?**: 사용자는 정해진 필드 외에 "매운 건 죽어도 싫어요", "분위기만 좋으면 됨"처럼 자유롭게 말합니다.
    - **Goal**: `extra_text`에 숨겨진 강경한 어조("절대", "Naver")와 유연한 어조("희망", "좋음")를 구별해내는 독해력을 봅니다.

### 3. Persona Quality (페르소나 퀄리티)
생성된 페르소나가 토론에서 얼마나 사람처럼 자연스럽게 행동할지 평가합니다.

- **`discussion_readiness` (토론 준비도)**
    - **Why?**: 생성된 시스템 프롬프트가 에이전트에게 "너는 누구고, 무엇은 절대 양보하면 안 되며, 무엇은 타협해도 되는지" 명확한 지침(Instruction)을 줘야 합니다.
    - **Goal**: 에이전트가 토론에 바로 투입될 수 있는 수준의 'Actionable Prompt' 인지 평가합니다.

- **`specificity` (구체성)**
    - **Why?**: "한식 좋아함"보다는 "얼큰한 김치찌개 같은 한식을 좋아함"이 훨씬 몰입감 있는 토론을 만듭니다.
    - **Goal**: 사용자의 구체적인 키워드를 뭉뚱그리지 않고 자세히 반영했는지 봅니다.

- **`reasoning_depth` (판단 근거)**
    - **Why?**: 모델이 왜 그런 페르소나를 만들었는지 설명할 수 있어야(XAI), 나중에 문제가 생겼을 때 디버깅이 가능합니다.

---

## 🧪 Test Dataset Strategy
To rigorously test the models, we designed **20 Diverse Scenarios** divided into 5 strategic categories. This ensures that high scores are not just due to memorization but true understanding.

| Category | Cases | Description | Key Evaluation Metrics |
|----------|-------|-------------|------------------------|
| **1. Normal Cases** | 3 | Simple, clear inputs with no conflicts. | `json_schema_compliance`, `field_coverage` (Baseline) |
| **2. Logic & Contradiction** | 5 | Inputs containing conflicting info (e.g., "Allergic to shellfish" but "Loves Crab"). | `consistency`, `classification_accuracy` (Safety Check) |
| **3. Ambiguity & Creativity** | 4 | Vague inputs like "Anywhere is fine" or "Hip vibes". | `extra_text_parsing`, `specificity` (User Intent understanding) |
| **4. Social Context** | 4 | Situational constraints like "Business Dinner", "Blind Date", "Foreigner Friend". | `discussion_readiness` (Contextual Awareness) |
| **5. Constraint Overload** | 4 | 5+ Allergies or extremely picky eaters. | `field_coverage` (Memory Capacity & Attention) |

This dataset is designed to expose the **"Intelligence Gap"** between simple models (7B) and advanced models (GPT-4 class).

---

## 🚀 How to Run

### 1. Environment Setup
```bash
# Clone Repository
git clone https://github.com/KT20201224/persona_test.git
cd persona_test

# Create .env file
cp .env.example .env
vi .env
```

### 2. Configure Models (`.env`)
```ini
# A. OpenAI API 사용 시
OPENAI_API_KEY=sk-...

# B. Hugging Face 자동 다운로드 사용 시 (추천)
TARGET_HF_MODELS=Qwen/Qwen2.5-3B-Instruct,yanolja/EEVE-Korean-10.8B-v1.0

# C. Langfuse Observability (선택)
LANGFUSE_SECRET_KEY=...
```

### 3. Run Evaluation (Docker)
```bash
docker compose up --build
```

### 4. Check Reports
실행 완료 후 `./reports` 폴더에 HTML 리포트가 생성됩니다.
- 📊 **Overall Score Comparison**: 모델별 종합 점수
- 🕷️ **Radar Chart**: 모델별 강점/약점 분석
- ⏱️ **Latency & TTFT**: 속도 및 효율성 분석
