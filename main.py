import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import LLMChain
import httpx
from enum import Enum
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import json
from fastapi.middleware.cors import CORSMiddleware
import asyncio  # 비동기 처리를 위한 라이브러리

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

CLOVA_SECRET = os.getenv('CLOVA_SECRET')
CLOVA_INVOKE_URL = os.getenv('CLOVA_INVOKE_URL')


app = FastAPI(title="Interview Q&A Service")

class JobType(str, Enum):
    ENGINEERING = "엔지니어링·설계"
    DEVELOPMENT = "개발·데이터"
    SALES = "영업"
    MANUFACTURING = "제조·생산"
    PLANNING = "기획·전략"

JOB_TYPE_MAPPING = {
    "엔지니어링·설계": ["엔지니어링·설계"],
    "개발·데이터": ["개발·데이터"],
    "영업": ["영업"],
    "제조·생산": ["제조·생산"],
    "기획·전략": ["기획,전략"]
}

class ResumeInput(BaseModel):
    resume_items: List[str] = Field(..., min_items=3, max_items=5, description="자기소개서 항목들")
    job_type: JobType = Field(..., description="직무 타입")

    @field_validator('resume_items')
    def validate_resume_items(cls, v):
        if not all(isinstance(item, str) and item.strip() for item in v):
            raise ValueError("모든 자기소개서 항목은 비어있지 않은 문자열이어야 합니다.")
        return v

class InterviewQuestions(BaseModel):
    questions: List[str] = Field(..., min_items=3, max_items=3)
    
    @field_validator('questions')
    def validate_questions(cls, v):
        # Ensure we have exactly 3 questions
        if len(v) != 3:
            raise ValueError("질문은 정확히 3개여야 합니다.")
        
        # Clean up each question
        cleaned_questions = []
        for i, question in enumerate(v, 1):
            # Remove any existing numbering and clean up whitespace
            q = question.strip()
            if q[0].isdigit() and '. ' in q[:4]:
                q = q[q.index('. ') + 2:]
            
            # Remove any newlines within the question
            q = q.replace('\n', ' ').strip()
            
            # Add proper numbering
            cleaned_questions.append(f"{i}. {q}")
            
        return cleaned_questions

def initialize_service():
    try:
        # Pinecone 초기화
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Embedding 모델 초기화
        embedding = UpstageEmbeddings(
            model="solar-embedding-1-large",
            api_key=UPSTAGE_API_KEY
        )
        
        # Vector Store 초기화
        database = PineconeVectorStore.from_existing_index(
            index_name='question-final',
            embedding=embedding
        )
        
        # LLM 초기화
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )

        # 프롬프트 템플릿 수정 - 쉼표로 구분된 응답 강조
        prompt = ChatPromptTemplate.from_template("""
다음은 사용자가 입력한 자기소개서입니다:
    
{resume}
    
지원자의 직무는 {job_type}입니다.
    
다음은 이 직무와 관련된 기존 질문들입니다:
{filtered_results}

위의 정보를 참고하여, 이 자기소개서를 바탕으로 면접관이 물어볼 만한 3개의 꼬리질문을 생성해주세요.
질문을 생성할 때 다음 사항을 고려해주세요:
1. 자기소개서의 내용을 더 깊이 파고들 수 있는 질문
2. 지원자의 경험이나 역량을 더 자세히 알아볼 수 있는 질문
3. 지원자의 동기나 열정을 확인할 수 있는 질문
4. 위에 제시된 기존 질문들을 참고하되, 자기소개서의 내용에 맞게 변형하거나 새로운 질문을 만들어주세요.

중요: 질문 각각은 반드시 완성된 문장으로 작성해야 하며, 각 질문은 쉼표 `;` 기호로만 구분해주세요. 정확히 3개의 질문을 생성하세요.
""")
        # Chain 설정
        chain = prompt | llm | CommaSeparatedListOutputParser()
        
        return database, chain
    except Exception as e:
        print(f"Service initialization failed: {str(e)}")
        raise

# 서비스 초기화
database, chain = initialize_service()
dictionary = ["회사명이 문장에 들어갈 경우 -> 회사"]

def get_related_questions(job_type: str, resume_text: str) -> str:
    try:
        job_type_values = JOB_TYPE_MAPPING.get(job_type, [job_type])
        filter_dict = {"job_type": {"$in": job_type_values}}
        
        results = database.similarity_search_with_score(
            resume_text,
            k=100,
            filter=filter_dict
        )
        
        filtered_results = [doc.page_content for doc, score in results][:4]
        return "\n".join(filtered_results)
    except Exception as e:
        print(f"Error in getting related questions: {str(e)}")
        return ""
    
def generate_questions(resume_items: List[str], job_type: JobType) -> List[str]:
    try:
        # Combine resume items
        combined_resume = "\n\n".join([f"문항 {i+1}:\n{item}" for i, item in enumerate(resume_items)])

        # Get related questions
        related_questions = get_related_questions(job_type.value, combined_resume)

        # Generate new questions using chain
        result = chain.invoke({
            "resume": combined_resume,
            "job_type": job_type.value,
            "filtered_results": related_questions,
            "dictionary": dictionary
        })

        # Handle both list and string results
        if isinstance(result, list):
            raw_text = ' '.join(result)
        else:
            raw_text = result

        # Split by both semicolon and number patterns
        import re
        
        # First, clean up any existing numbering that might be in the middle of text
        cleaned_text = re.sub(r'\s+\d+\.\s+', '; ', raw_text)
        
        # Split by semicolon
        raw_questions = [q.strip() for q in cleaned_text.split(';') if q.strip()]
        
        # Clean up questions:
        # 1. Remove any remaining numbers at start
        # 2. Remove any fragments that look like partial questions
        cleaned_questions = []
        for q in raw_questions:
            # Remove leading numbers and dots
            q = re.sub(r'^\d+\.\s*', '', q.strip())
            
            # Only add if it looks like a complete question (ends with punctuation or is long enough)
            if q.endswith('?') or q.endswith('.') or len(q) > 30:
                # Ensure it ends with a question mark if it doesn't have ending punctuation
                if not any(q.endswith(p) for p in '.?!'):
                    q += '?'
                cleaned_questions.append(q)

        # Ensure exactly 3 questions
        while len(cleaned_questions) < 3:
            cleaned_questions.append("자기소개서 내용에 대해 더 자세히 설명해주실 수 있습니까?")
        cleaned_questions = cleaned_questions[:3]  # Truncate to 3 questions if more

        # Format with numbers
        formatted_questions = [f"{i}. {q}" for i, q in enumerate(cleaned_questions, 1)]

        return formatted_questions
        
    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to generate questions: {str(e)}")

#stt
class VideoRequest(BaseModel):
    video: str # 입력 형식: {"video": "비디오URL"}

class STTResponse(BaseModel):
    status: str # 처리 상태
    stt_text: str # 변환된 텍스트
    ai_analysis: Optional[dict] = None # AI 분석 결과 (선택적)

class ClovaSpeechClient:
    def __init__(self):
        self.invoke_url = CLOVA_INVOKE_URL
        self.secret = CLOVA_SECRET

    async def transcribe_url(self, url: str) -> str:
        request_body = {
            'url': url,
            'language': 'ko-KR', # 한국어 인식
            'completion': 'sync' # 동기 방식 처리
        }
        
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        
        # httpx를 사용한 비동기 HTTP 요청
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=self.invoke_url + '/recognizer/url',
                headers=headers,
                json=request_body
            )

            print(f"Clova API Response: {response.text}")  # 응답 확인
            
            result = response.json()
            text = result.get('text', '')
            
            if not text:
                print(f"Empty text result. Full response: {result}")  # 전체 응답 확인
                
            return text

class AIProcessor:
    def __init__(self):
        self.api1_endpoint = "http://43.201.48.59:8000/answer_predict"
        self.api2_endpoint = "http://43.203.197.157:8000/combined-feedback"
    
    async def _call_api(self, endpoint: str, payload: dict) -> None:
        """
        개별 API 호출 비동기 메서드
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url=endpoint, json=payload, timeout=30.0)
                if response.status_code != 200:
                    print(f"API 호출 실패: {endpoint}, 응답 코드: {response.status_code}")
                else:
                    print(f"API 호출 성공: {endpoint}")
            except Exception as e:
                print(f"API 호출 중 오류 발생: {endpoint}, 오류: {str(e)}")

    async def process_text(self, text: str) -> None:
        """
        텍스트를 두 AI API에 병렬로 전송
        """
        payload = {"answer": text}
        
        # 두 API 호출을 병렬로 실행
        tasks = [
            self._call_api(self.api1_endpoint, payload),
            self._call_api(self.api2_endpoint, payload)
        ]
        
        await asyncio.gather(*tasks)  # 두 작업을 병렬로 실행

                
# FastAPI 앱 설정 (기존 app 사용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stt_client = ClovaSpeechClient()
ai_processor = AIProcessor()  
database, chain = initialize_service()


@app.post("/generate-questions", response_model=InterviewQuestions)
async def generate_interview_questions(resume_input: ResumeInput) -> InterviewQuestions:
    try:
        questions = generate_questions(
            resume_items=resume_input.resume_items,
            job_type=resume_input.job_type
        )
        return InterviewQuestions(questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/process-video", response_model=STTResponse)
async def process_video(request: VideoRequest):
    try:
        #  1. VideoRequest에서 video URL 추출 후 STT 처리
        text = await stt_client.transcribe_url(request.video)
        
        # 2. AI 처리 - 변환된 텍스트에 대한 AI 분석 수행
        await ai_processor.process_text(text)
        
        return STTResponse(
            status="success",
            stt_text=text        )
    
    except Exception as e:
        print(f"Process Video Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/test-stt")
async def test_stt():
    """STT 기능 테스트를 위한 엔드포인트"""
    test_video_url = "https://capstone-viewit.s3.ap-northeast-2.amazonaws.com/%E1%84%8F%E1%85%A2%E1%86%B8%E1%84%83%E1%85%B5+%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%90%E1%85%A6%E1%84%89%E1%85%B3%E1%84%90%E1%85%B3%E1%84%8B%E1%85%AD%E1%86%BC.mp4"
    try:
        # API 응답 전체를 로깅
        text = await stt_client.transcribe_url(test_video_url)
        print(f"STT Response: {text}")  # 응답 확인용 로그
        
        if not text:
            return {
                "status": "error",
                "detail": "STT 변환 결과가 비어있습니다."
            }
            
        return {"status": "success", "test_text": text}
    except Exception as e:
        print(f"STT Error: {str(e)}")  # 에러 로깅
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)