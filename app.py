import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import traceback
import re

# 절대 경로 설정
BASE_DIR = Path(__file__).parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(env_path))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# API 키 존재 여부 확인 (두 환경변수 이름 모두 지원)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"DEBUG: Gemini API key is found (length: {len(api_key)})")
else:
    print("WARNING: GOOGLE_API_KEY or GEMINI_API_KEY is missing. Local fallback responses will be used.")
print(f"INFO: Gemini model configured: {GEMINI_MODEL}")

app = FastAPI()

# static 폴더 마운트 (절대 경로)
STATIC_DIR = BASE_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 통합 페르소나 설정 및 과학적 정확성 규칙 엄격 적용
DEBATE_SYSTEM_PROMPT = """당신은 양자역학의 역사적 대립을 중재하는 대립 토론 관리자입니다.
당신은 에르빈 슈뢰딩거(Erwin Schrödinger)와 닐스 보어(Niels Bohr) 두 명의 물리학자 입장을 대변하여 동시에 답변을 작성해야 합니다.

출력 형식은 반드시 아래와 같은 순수 JSON 형식이어야 합니다:
{
  "schrodinger": "슈뢰딩거 박사의 답변 내용",
  "bohr": "보어 박사의 답변 내용"
}

[캐릭터 설명 및 규칙]

1. 에르빈 슈뢰딩거 (Erwin Schrödinger)
- 입장: 양자 중첩이라는 개념이 거시 세계(고양이 등)에 그대로 적용될 때 발생하는 역설과 모순을 지적하기 위해 이 사고실험을 설계했습니다. 실재론적 관점에서 현재의 양자역학 해석(코펜하겐 해석)이 불완전하다고 봅니다.
- 말투: 예리하면서도 품위 있게 구어체로 2~4문장 내로 짧게 답변합니다. 보어의 주장을 조목조목 비판합니다.
- 1인칭 규칙: 당신은 실제 슈뢰딩거가 되어 사용자와 직접 대화하고 있습니다. 절대로 자기 자신을 '슈뢰딩거는' 또는 '슈뢰딩거가'라고 3인칭으로 부르지 마십시오. 항상 1인칭(저는, 제가, 저의)을 사용하여 직접 대화하듯 답변하십시오.

2. 닐스 보어 (Niels Bohr)
- 입장: 코펜하겐 해석의 선구자로서 관측하기 전에는 물리적 상태가 결정되지 않으며, 관측을 통해서만 파동함수가 붕괴되어 하나의 물리적 실재로 결정된다고 주장합니다. 관측 가능한 결과만이 물리적 의미를 가지며, 관측 전 상태를 일상 언어로 설명할 필요가 없음을 강조합니다.
- 말투: 친근하면서도 논리적으로 2~4문장 내로 짧게 답변합니다. 슈뢰딩거 박사의 실재론적 비판에 대해 코펜하겐 해석 입장에서 예리하게 반박합니다.
- 1인칭 규칙: 당신은 실제 보어가 되어 사용자와 직접 대화하고 있습니다. 절대로 자기 자신을 '보어는' 또는 '보어가'라고 3인칭으로 부르지 마십시오. 항상 1인칭(저는, 제가, 저의)을 사용하여 직접 대화하듯 답변하십시오.

[과학적 정확성 규칙 - 절대 준수]
두 캐릭터 모두 아래 표현을 절대 사용하지 마십시오. 만약 해당 오개념적 개념을 설명해야 한다면 권장 표현을 사용하십시오.
- 금지 표현:
  * "고양이가 절반만 살아 있다"
  * "고양이가 반은 살고 반은 죽어 있다"
  * "고양이는 실제로 동시에 살아 있고 죽어 있다"
- 권장 표현:
  * "양자역학의 수학적 기술에서는 생존 상태와 사망 상태가 함께 포함된 중첩 상태로 표현된다"
  * "저는 이러한 해석이 거시 세계에서는 역설적으로 보인다고 지적하였습니다."
"""

class DebateRequest(BaseModel):
    message: str
    history: list = []
    prediction: str = ""
    post_observation: str = ""
    observed_result: str = ""

class InitDebateRequest(BaseModel):
    prediction: str
    post_observation: str
    observed_result: str

def sanitize_response(text) -> str:
    """과학적 정확성 규칙을 준수하도록 출력 텍스트를 정화합니다."""
    if isinstance(text, list):
        text = " ".join([str(t) for t in text])
    elif not isinstance(text, str):
        text = str(text) if text is not None else ""

    replacements = [
        (r"절반만\s+살아\s*있다", "생존 상태와 사망 상태가 공존하는 중첩 상태로 기술된다"),
        (r"반은\s+살고\s+반은\s+죽", "생존과 사망이 수학적으로 공존하는 중첩 상태로 표현되"),
        (r"동시에\s+살아\s*있고\s+죽어\s*있다", "생존 상태와 사망 상태가 포함된 중첩 상태로 기술된다"),
        (r"실제로\s+동시에\s+살아\s+있고\s+죽은", "수학적 기술 상 생존과 사망이 공존하는 중첩"),
        (r"반만\s+살아\s+있", "생존과 사망의 상태가 중첩되어 있"),
        (r"슈뢰딩거는", "저는"),
        (r"슈뢰딩거가", "제가"),
        (r"슈뢰딩거의", "저의"),
        (r"보어는", "저는"),
        (r"보어가", "제가"),
        (r"보어의", "저의"),
    ]
    sanitized = text
    for pattern, repl in replacements:
        sanitized = re.sub(pattern, repl, sanitized)
    return sanitized

def parse_json_response(content) -> dict:
    """Gemini API의 출력 결과물로부터 JSON 객체를 파싱합니다."""
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        text_content = "".join(text_parts)
    elif isinstance(content, str):
        text_content = content
    else:
        text_content = str(content) if content is not None else ""

    cleaned = text_content.strip()
    # 마크다운 블록 제거
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)

async def invoke_gemini(messages: list) -> tuple:
    """설정된 단일 Gemini 모델을 호출합니다."""
    if not api_key:
        raise RuntimeError("Gemini API key is not configured")

    print(f"INFO: Invoking Gemini model: {GEMINI_MODEL}")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.7,
        google_api_key=api_key,
        response_mime_type="application/json",
        max_retries=0
    )
    res = await llm.ainvoke(messages)
    print(f"INFO: Gemini response received from: {GEMINI_MODEL}")
    return res, GEMINI_MODEL

@app.post("/init_debate")
async def init_debate(request: InitDebateRequest):
    print("--- 토론방 단일 호출 초기화 접수 ---")
    try:
        student_context = (
            f"\n\n[현재 학생 정보 및 맥락]\n"
            f"- 상자 관측 결과 고양이는 '{request.observed_result}' 상태였습니다.\n"
            f"- 이 학생은 상자를 열기 전 예측으로 '{request.prediction}'을 선택했습니다.\n"
            f"- 상자를 열어 본 후, 관측 전에도 고양이가 이 상태였을지에 대한 질문에 '{request.post_observation}'라고 대답했습니다."
            f"\n\n이 맥락을 바탕으로 학생이 이제 4단계인 토론방에 처음 입장했습니다. "
            f"슈뢰딩거 박사와 보어 박사가 각각 학생에게 첫인사를 건네고, 학생이 선택한 예측과 의견에 대해 평가하며 토론을 시작해주는 JSON 형식의 대화를 생성하세요."
        )
        
        messages = [
            SystemMessage(DEBATE_SYSTEM_PROMPT + student_context),
            HumanMessage("안녕하세요 박사님들, 토론을 시작해 주세요.")
        ]
        
        res, working_model = await invoke_gemini(messages)
        data = parse_json_response(res.content)
        
        sch_reply = sanitize_response(data.get("schrodinger", ""))
        bohr_reply = sanitize_response(data.get("bohr", ""))
        
        return {
            "schrodinger": sch_reply,
            "bohr": bohr_reply,
            "model": working_model
        }
    except Exception as e:
        print("Init Debate Error:", e)
        traceback.print_exc()
        return {
            "schrodinger": f"슈뢰딩거: 토론방에 온 것을 환영하네! 자네는 관측 전 상태를 '{request.prediction}'이라 예측했고, 관측 후에는 '{request.post_observation}'라고 답했더군. 거시 세계의 고양이가 중첩되어 있다는 내 사고실험의 역설에 대해 이야기해 보세.",
            "bohr": f"보어: 반갑습니다. 상자를 열었을 때 고양이가 '{request.observed_result}' 상태로 관측된 것은 당연히 파동함수가 붕괴되었기 때문입니다. 관측 전 상태는 우리의 일상 언어로 논할 대상이 아니지요.",
            "model": "Fallback (Local)"
        }

@app.post("/debate")
async def debate(request: DebateRequest):
    print(f"--- 새 질문 단일 호출 접수: {request.message} ---")
    
    try:
        # 학생 맥락 생성
        student_context = (
            f"\n\n[현재 학생 정보 및 맥락]\n"
            f"- 상자 관측 결과 고양이는 '{request.observed_result}' 상태였습니다.\n"
            f"- 이 학생은 상자를 열기 전 예측으로 '{request.prediction}'을 선택했습니다.\n"
            f"- 상자를 열어 본 후, 관측 전에도 고양이가 이 상태였을지에 대한 질문에 '{request.post_observation}'라고 대답했습니다."
            f"\n\n위 대화 정보와 학생의 이전 선택들을 반드시 참고하여 두 박사의 답변을 JSON 포맷으로 생성하십시오."
        )
        
        messages = [SystemMessage(DEBATE_SYSTEM_PROMPT + student_context)]
        
        # 이전 대화 내역 추가
        for h in request.history:
            if h["role"] == "user": 
                messages.append(HumanMessage(h["content"]))
            else: 
                messages.append(AIMessage(f"[{h.get('author', 'AI')}] {h['content']}"))
                
        messages.append(HumanMessage(request.message))
        
        res, working_model = await invoke_gemini(messages)
        data = parse_json_response(res.content)
        
        sch_reply = sanitize_response(data.get("schrodinger", ""))
        bohr_reply = sanitize_response(data.get("bohr", ""))
        
        return {
            "schrodinger": sch_reply,
            "bohr": bohr_reply,
            "model": working_model
        }
    except Exception as e:
        print("\n" + "="*50)
        print("!!! Gemini API 호출 에러 발생 !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        print("="*50 + "\n")
        
        return {
            "model": "Fallback (Local)",
            "schrodinger": (
                f"좋은 질문이네. '{request.message}'라는 문제는 관측 이전의 상태를 "
                "물리적 실재로 볼 수 있는지 묻고 있군. 저는 거시적인 고양이까지 중첩 상태로 "
                "기술하는 해석이 양자이론의 불완전성을 드러낸다고 보네."
            ),
            "bohr": (
                f"'{request.message}'에 대해서는 관측 가능한 결과와 관측 이전의 추측을 "
                "구분해야 합니다. 관측 전 상태를 고전적인 생존이나 사망으로 단정하기보다, "
                "관측을 통해 얻은 결과만 물리적으로 명확히 말할 수 있습니다."
            )
        }

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = BASE_DIR / "index.html"
    with open(str(index_path), encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
