import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# 절대 경로 설정
BASE_DIR = Path(__file__).parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(env_path))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("API 키가 설정되지 않았습니다.")
else:
    genai.configure(api_key=api_key)
    print(f"설정된 Gemini 모델: {GEMINI_MODEL}")
    print("사용 가능한 모델 목록:")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name.removeprefix("models/"))
            print(f"- {m.name} ({m.display_name})")

    if GEMINI_MODEL in available_models:
        print(f"확인 완료: {GEMINI_MODEL} 모델을 사용할 수 있습니다.")
    else:
        print(f"주의: {GEMINI_MODEL} 모델이 generateContent 목록에서 확인되지 않았습니다.")
