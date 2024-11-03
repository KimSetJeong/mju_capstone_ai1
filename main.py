import os
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from enum import Enum
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, PodSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Validate required environment variables
if not all([OPENAI_API_KEY, UPSTAGE_API_KEY, PINECONE_API_KEY]):
    raise ValueError("Required environment variables are missing. Please check your .env file.")

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

    @validator('resume_items')
    def validate_resume_items(cls, v):
        if not all(isinstance(item, str) and item.strip() for item in v):
            raise ValueError("모든 자기소개서 항목은 비어있지 않은 문자열이어야 합니다.")
        return v

class InterviewQuestions(BaseModel):
    questions: List[str]

PROMPT_TEMPLATE = """
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

우리의 사전을 참고하여 필요한 경우 용어를 변경해주세요:
사전: {dictionary}

질문들을 쉼표로 구분하여 리스트 형태로 출력해주세요.
"""

class InterviewService:
    def __init__(self, llm, database):
        self.llm = llm
        self.database = database
        self.dictionary = ["회사명이 문장에 들어갈 경우 -> 회사"]
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.output_parser = CommaSeparatedListOutputParser()
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.output_parser
        )

    def get_related_questions(self, job_type: str, resume: str) -> str:
        search_terms = JOB_TYPE_MAPPING[job_type]
        
        all_results = []
        for term in search_terms:
            results = self.database.similarity_search_with_score(resume, k=50)
            filtered_results = [
                (doc.page_content, score) for doc, score in results
                if f"job_type: {term}" in doc.page_content
            ]
            all_results.extend(filtered_results)
        
        sorted_results = sorted(all_results, key=lambda x: x[1])
        top_results = [content for content, _ in sorted_results[:4]]
        
        if not top_results:
            results = self.database.similarity_search_with_score(resume, k=4)
            top_results = [doc.page_content for doc, _ in results]
        
        return "\n".join(top_results)

    def generate_questions(self, resume_items: List[str], job_type: JobType) -> List[str]:
        combined_resume = "\n\n".join([f"문항 {i+1}:\n{item}" for i, item in enumerate(resume_items)])
        
        related_questions = self.get_related_questions(job_type.value, combined_resume)
        result = self.chain.invoke({
            "resume": combined_resume,
            "job_type": job_type.value,
            "filtered_results": related_questions,
            "dictionary": self.dictionary
        })
        
        questions = [q.strip() for q in result if q.strip()]
        return questions

def initialize_service():
    try:
        # Pinecone 클라이언트 초기화
        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )
        
        # UpstageEmbeddings 설정
        embedding = UpstageEmbeddings(
            model="solar-embedding-1-large",
            api_key=UPSTAGE_API_KEY
        )
        
        # 인덱스 초기화
        index = pc.Index('question-final')
        
        # Pinecone 벡터 스토어 초기화
        database = PineconeVectorStore(
            index=index,
            embedding=embedding
        )
        
        # ChatGPT 모델 설정
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        return InterviewService(llm=llm, database=database)
    except Exception as e:
        print(f"Service initialization failed: {str(e)}")
        raise

interview_service = initialize_service()

@app.post("/generate-questions", response_model=InterviewQuestions)
async def generate_interview_questions(resume_input: ResumeInput) -> InterviewQuestions:
    try:
        questions = interview_service.generate_questions(
            resume_items=resume_input.resume_items,
            job_type=resume_input.job_type
        )
        return InterviewQuestions(questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)