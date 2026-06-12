# 슈뢰딩거 고양이 사고실험 시뮬레이션

## 프로젝트 소개

양자역학의 대표적인 사고실험인 슈뢰딩거의 고양이를 학생들이 직접 체험하고 토론할 수 있도록 제작한 웹 기반 시뮬레이션입니다.

### 주요 기능

* 사고실험 단계별 체험
* 상자 관측 및 결과 확인
* 닐스 보어 vs 에르빈 슈뢰딩거 토론
* 투표 및 성찰 질문
* 학습 데이터 다운로드

---

## 사용 기술

* FastAPI
* HTML / CSS / JavaScript
* Gemini API
* Render

---

## 로컬 실행

### 패키지 설치

```bash
pip install -r requirements.txt
```

### 환경변수 설정

`.env` 파일 생성

```env
GEMINI_API_KEY=본인의_API_KEY
```

### 서버 실행

```bash
uvicorn app:app --reload
```

접속

```text
http://127.0.0.1:8000
```

---

## Render 배포

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Environment Variable

```env
GEMINI_API_KEY=본인의_API_KEY
```

---

## GitHub 저장소

https://github.com/tngus2914-alt/schrodinger_debate

## 배포 주소

https://schrodinger-debate.onrender.com

## Gemini 모델 설정

기본 모델은 `gemini-3.5-flash`입니다.

로컬 `.env`와 Render의 Environment 설정에 다음 값을 사용하세요.

```env
GEMINI_MODEL=gemini-3.5-flash
```

서버 시작 로그의 `INFO: Gemini model configured`와 API 호출 로그의
`INFO: Invoking Gemini model`에서 실제 사용 모델을 확인할 수 있습니다.
