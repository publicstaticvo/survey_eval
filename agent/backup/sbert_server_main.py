from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Union
import uvicorn
import numpy as np

# 全局加载模型（只加载一次）
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型"""
    global model
    print("正在加载模型...")
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
    print("模型加载完成！")
    yield


# 初始化 FastAPI 应用
app = FastAPI(
    title="Sentence Transformer API",
    description="使用 ms-marco-MiniLM-L12-v2 模型进行文本编码",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class TextRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="单个文本或文本列表")
    normalize: bool = Field(True, description="是否归一化向量")


class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="第一个文本")
    text2: str = Field(..., description="第二个文本")


# 响应模型
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int


class SimilarityResponse(BaseModel):
    similarity: float
    text1: str
    text2: str


# @app.on_event("startup")
# async def load_model():
#     print("正在加载模型...")
#     model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
#     print("模型加载完成！")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Sentence Transformer API",
        "model": "ms-marco-MiniLM-L12-v2",
        "endpoints": {
            "encode": "/encode",
            "similarity": "/similarity",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "healthy", "model_loaded": True}


@app.post("/encode", response_model=EmbeddingResponse)
async def encode_texts(request: TextRequest):
    """
    对文本进行编码，返回向量表示
    
    - **texts**: 单个文本字符串或文本列表
    - **normalize**: 是否对向量进行 L2 归一化
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 处理输入
        texts = [request.texts] if isinstance(request.texts, str) else request.texts
        
        # 生成嵌入向量
        embeddings = model.encode(
            texts,
            normalize_embeddings=request.normalize,
            convert_to_numpy=True
        )
        
        # 转换为列表格式
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0]),
            count=len(embeddings_list)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"编码错误: {str(e)}")


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    计算两个文本之间的余弦相似度
    
    - **text1**: 第一个文本
    - **text2**: 第二个文本
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 生成嵌入向量
        embeddings = model.encode(
            [request.text1, request.text2],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # 计算余弦相似度
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        
        return SimilarityResponse(
            similarity=similarity,
            text1=request.text1,
            text2=request.text2
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"相似度计算错误: {str(e)}")


@app.get("/model-info")
async def model_info():
    """获取模型信息"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_name": "msmarco-MiniLM-L-12-v3",
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
    }


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "main:app",  # 假设文件名为 main.py
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境设置为 False
        workers=1  # 可以根据需要调整工作进程数
    )