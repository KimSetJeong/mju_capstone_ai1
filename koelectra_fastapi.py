import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psutil
import os
from typing import Dict
import gc
import logging
from transformers import ElectraConfig
import time
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic 모델 정의 : 요청 데이터를 정의하는 모델
class PredictionRequest(BaseModel):
    answer: str #사용자가 예측을 원하는 텍스트를 포함함

class PredictionResponse(BaseModel):
    top3_predictions: Dict[str, float]

# swap 메모리 관리 함수 : 시스템 메모리가 일정 임계치(기본 75%)를 초과하면 메모리 관리를 수행
def optimize_swap_memory():
    """Swap 메모리 최적화 함수"""
    try:
        # swappiness 값을 높여서 swap 사용을 촉진 (기본값은 보통 60)
        os.system('sudo sysctl -w vm.swappiness=80')
        
        # swap 메모리 현재 상태 확인
        swap = psutil.swap_memory()
        print(f"Swap 사용량: {swap.percent}%")
        
        if swap.percent > 80:  # swap 사용량이 80% 이상이면
            print("Swap 메모리 정리 시작...")
            # swap 메모리 비우고 다시 활성화
            os.system('sudo swapoff -a')
            os.system('sudo swapon -a')
            print("Swap 메모리 정리 완료")
            
        return True
    except Exception as e:
        print(f"Swap 메모리 최적화 중 에러 발생: {str(e)}")
        return False
    
def manage_memory(threshold_percent=75):
    """메모리 관리 개선 함수"""
    
    memory = psutil.virtual_memory()
    if memory.percent > threshold_percent:
        print("메모리 정리 시작...")
        
        # Python 가비지 컬렉터 실행
        gc.collect()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Linux 시스템에서의 추가 메모리 최적화
        if os.name == 'posix':
            try:
                # 파일시스템 캐시 동기화
                os.system('sync')
                
                # 페이지 캐시 정리
                os.system('echo 3 > /proc/sys/vm/drop_caches')
                
                # swap 메모리 최적화
                optimize_swap_memory()
                
            except Exception as e:
                print(f"메모리 최적화 중 에러 발생: {str(e)}")
        
        print("메모리 정리 완료")
   


class EmotionPredictor:
    def __init__(self, model_path):
        print("모델 초기화 시작...")
        start_time = time.time()
        
        # 초기 메모리 상태 확인 및 최적화
        print("\n초기 메모리 상태:")
        manage_memory(threshold_percent=70)
        
        # GPU/CPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n사용 디바이스: {self.device}")
        
        print("\n1/4 단계: 레이블 매핑 설정")
        self.label_mapping = {
            "가치관": 0, "갈등 관리": 1, "개선능력": 2, "개인 정보": 3, "개인성": 4,
            "고객신념": 5, "고객지향": 6, "대처 능력": 7, "도전정신": 8, "설비": 9,
            "일정관리": 10, "적응력": 11, "전문성": 12, "직무 진정성": 13,
            "직업 가치관": 14, "트렌디함": 15, "해석력": 16, "행정처리": 17,
            "협동성": 18, "협업 진정성": 19
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        print("\n2/4 단계: 토크나이저 로드 중...")
        # 메모리 최적화를 위해 토크나이저 로드 전 메모리 정리
        manage_memory()
        self.tokenizer = ElectraTokenizer.from_pretrained(
            "monologg/koelectra-base-v3-discriminator",
            model_max_length=256,
            local_files_only=True
        )
        
        print("\n3/4 단계: 모델 구성 중...")
        # 메모리 최적화를 위해 모델 구성 전 메모리 정리
        manage_memory()
        config = ElectraConfig.from_pretrained(
            "monologg/koelectra-base-v3-discriminator",
            num_labels=len(self.label_mapping),
            problem_type="single_label_classification"
        )
        self.model = ElectraForSequenceClassification(config)
        
        print("\n4/4 단계: 모델 가중치 로드 중...")
        # 메모리 최적화를 위해 가중치 로드 전 메모리 정리
        manage_memory()
        print(f"가중치 파일 로드 중: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if "electra.embeddings.position_ids" in checkpoint:
            del checkpoint["electra.embeddings.position_ids"]
        
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 불필요한 메모리 정리
        del checkpoint
        manage_memory()
        
        end_time = time.time()
        print(f"\n모델 로딩 완료! 소요 시간: {end_time - start_time:.2f}초")
        print("\n최종 메모리 상태:")

    @torch.no_grad()
    def predict(self, text):
        # 예측 전 메모리 정리
        logger.info(f"예측 시작: {text[:50]}...")  # 텍스트 앞부분만 로깅
        manage_memory()
        
        try:
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            top3_probs, top3_indices = torch.topk(predictions[0], 3)
            
            top3_results = {
                self.reverse_mapping[idx.item()]: round(prob.item() * 100, 2)
                for prob, idx in zip(top3_probs, top3_indices)
            }

            logger.info(f"예측 결과: {top3_results}")
            
            # 메모리 정리
            del outputs, predictions, inputs
            manage_memory()
            
            return {'top3_predictions': top3_results}
            
        except Exception as e:
            print(f"예측 중 에러 발생: {str(e)}")
            raise

# FastAPI 앱 초기화
app = FastAPI(title="Emotion Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 예측기 초기화
predictor = None

@app.on_event("startup") #서버 시작 시 모델을 로드하여 전역 변수 predictor에 저장합니다.
async def startup_event():
    global predictor
    try:
        logger.info("서버 시작 중...")
        # 서버 시작 전 swap 메모리 최적화
        optimize_swap_memory()
        predictor = EmotionPredictor("best_model_weights_early_stopped.pt")
        logger.info("서버 시작 완료")
    except Exception as e:
        print(f"서버 시작 중 에러 발생: {str(e)}")
        raise

@app.post("/answer_predict", response_model=PredictionResponse) #예측을 수행하고 그 결과를 반환함
async def predict_emotion(request: PredictionRequest):
    logger.info("새로운 예측 요청 받음")
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        result = predictor.predict(request.answer)
        logger.info("예측 완료")
        return result
    except Exception as e:
        logger.error(f"예측 처리 중 에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("서버 시작 중...")
    import uvicorn

    optimize_swap_memory()

    uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,
            loop="asyncio",
            timeout_keep_alive=120  # Keep-alive 타임아웃 증가
    
        )