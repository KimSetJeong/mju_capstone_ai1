import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import LLMChain
from enum import Enum
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)