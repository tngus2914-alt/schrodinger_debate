import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# 절대 경로 설정
BASE_DIR = Path(__file__).parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(env_path))

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("API 키가 설정되지 않았습니다.")
else:
    genai.configure(api_key=api_key)
    print("사용 가능한 모델 목록:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name} ({m.display_name})")
