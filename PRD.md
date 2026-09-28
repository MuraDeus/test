# NAI V5 만화 생성기 (Mobile-Optimized Web App) - PRD & Technical Spec

## 1. 프로젝트 개요 및 목표
- **목적**: Galaxy S26 Ultra (Termux Python 환경)에서 실행되는 모바일 최적화 NAI V5 만화/웹툰 컷 생성 웹 애플리케이션
- **주요 스택**: Python 3.11+, Gradio (`0.0.0.0:7860`), NovelAI Diffusion V5 API, LLM API (Claude Sonnet 등)
- **데이터 지속성**: `config.json`, `presets.json`, `characters.json`, `outputs/` 내 모든 데이터 영구 보존

---

## 2. 모듈별 구조 및 역할 (`/` 루트 위치)
1. **`config_manager.py`**: 설정을 저장/조회/수정/삭제(CRUD)하는 데이터 지속성 관리 모듈
2. **`llm_parser.py`**:
   - Cloudflare WAF 1010 차단 방지용 `User-Agent` Header가 명시된 LLM API 호출 모듈
   - 상황/대사/문맥을 NAI V5 규격 프롬프트로 변환
3. **`nai_client.py`**: Base Prompt, Character Prompts, 대사 조건을 결합하여 NAI V5 API 호출 및 이미지 저장 모듈
4. **`app.py`**: Gradio 기반 모바일 최적화 5개 탭 UI (`0.0.0.0:7860`)
5. **`requirements.txt`**: Termux 설치용 필요 라이브러리 목록

---

## 3. 핵심 탭 구조 및 사양
1. **만화 생성 (Main Generation)**
   - 대본/상황 입력 + 대사 체크박스 & 한국어 대사 입력 칸
   - 빈칸 감지 기반 AI 자동 연속 생성 (`context.json` 문맥 참조)
   - 모델 추론 정도 (`None`, `Low`, `Medium`, `High`) 선택 드롭다운
2. **인물(Character) 프리셋 관리 (CRUD)**
   - 인물 목록 클릭 시 입력창 자동 채움, 새로 등록 / 수정 / 삭제 버튼 구현
3. **NAI 그림체 프리셋 관리 (CRUD)**
   - 그림체 목록 클릭 시 입력창 자동 채움, 새로 등록 / 수정 / 삭제 버튼 구현
4. **갤러리 및 그룹 관리**
   - 그룹별 이미지 Grid 뷰, 이미지 개별 삭제, 그룹 전체 삭제/이름 변경
5. **모델 및 API 설정**
   - LLM & NAI API Key, 모델명, LLM 기본 베이스 프롬프트(Base Prompt) 설정

---

## 4. NAI V5 프롬프트 & 대사 렌더링 스펙 규칙
- **공식 API 레퍼런스**: [NovelAI API Reference](https://docs.novelai.net/en/scripting/api-reference)
- **Multi-Character**: `Base Prompt | Character 1 | Character 2` (`characterPrompts` 배열 활용)
- **대사 처리**: Base Prompt 맨 마지막에 `speech bubble, text, korean text, Text: "대사"` 배치 (이후 태그 오염 방지)
