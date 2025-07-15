from fastapi import FastAPI, Header, HTTPException, Response, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import json
from configs.config import load_config
from configs.logging_config import setup_logging

from source.models.vector_indexing import MilvusVectorStore
from source.models.vector_search import MilvusVectorRetriever
from source.models.elastic_search import ProductElasticSQLRetriever
from configs.config import load_config
from configs.logging_config import setup_logging
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()
DIFY_API_KEY = os.getenv("DIFY_API_KEY")

logger = setup_logging("Chatbot_SaleForce")

config = load_config()
milvus_uri = config["milvus"]["milvus_uri"]
milvus_embedding_dimension = config["milvus"]["embedding_dimension"]
embedding_model_name = config["milvus"]["embedding_model"]

openai_key = config["llm"]["openai_api_key"]

milvus_embedding_model = OpenAIEmbeddings(
                        openai_api_key=openai_key,
                        model=embedding_model_name,
                        dimensions=milvus_embedding_dimension,
                    )

product_vectorstore = MilvusVectorStore(
    milvus_uri=milvus_uri,
    collection_name=config["milvus"]["product_collection_name"],
    embeddings=milvus_embedding_model,
    dimensions=milvus_embedding_dimension,
    openai_key=openai_key,
    recreate_collection=False,
    logger=logger,
).vectorstore
milvus_product_vector_retriever = MilvusVectorRetriever(product_vectorstore, logger)

service_vectorstore = MilvusVectorStore(
    milvus_uri=milvus_uri,
    collection_name=config["milvus"]["service_collection_name"],
    embeddings=milvus_embedding_model,
    dimensions=milvus_embedding_dimension,
    openai_key=openai_key,
    recreate_collection=False,
    logger=logger,
).vectorstore
milvus_service_vector_retriever = MilvusVectorRetriever(service_vectorstore, logger)


es_host = config['elasticsearch']['host']
es_username = config['elasticsearch']['Username']
es_password = config['elasticsearch']['Password']
es = Elasticsearch(es_host, basic_auth=(es_username, es_password), verify_certs=False)  

elastic_sql_retriever = ProductElasticSQLRetriever(es, logger)


# Pydantic models for request and response wwith vector search
class RetrievalSetting(BaseModel):
    top_k: int = 5
    score_threshold: float = 0

class RetrievalRequest(BaseModel):
    knowledge_id: str = None
    query: str
    retrieval_setting: RetrievalSetting

class RetrievalRecord(BaseModel):
    content: str
    title: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    
class RetrievalResponse(BaseModel):
    records: List[RetrievalRecord]

class SqlRetrievalRequest(BaseModel):
    query: str

class SQLRetrievalResponse(BaseModel):
    results: str
    sql_query: str

app = FastAPI(title="Retrieval API")

def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ")[1]
    if token != DIFY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return True
    
    
@app.get("/secure")
def secure_endpoint(_: bool = Depends(verify_api_key)):
    return {"message": "✅ Access granted"}

def filter_results_by_threshold(results: list, threshold: Optional[float] = None):
    if threshold is None:
        return results
    return [(doc, score) for doc, score in results if score >= threshold]

def convert_to_retrieval_records(filtered_results: list) -> list:
    records = []
    for doc, score in filtered_results:
        content = doc.page_content
        try:
            parsed_content = json.loads(content)
            title = parsed_content.get("name", "Chưa rõ tên")
        except Exception:
            title = "Chưa rõ tên"
        record = RetrievalRecord(
            content=content,
            score=score,
            title=title,
            metadata={}
        )
        records.append(record)
    return records

@app.post("/service_vector_retrieval", response_model=RetrievalResponse)
async def service_vector_retrieval(request: RetrievalRequest,
                                   _: str = Depends(verify_api_key)):
    if milvus_service_vector_retriever is None:
        logger.info("Lỗi: chưa khởi tạo service vectorstore retrieval")
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        results = milvus_service_vector_retriever.retrieve(
            query=request.query, 
            top_k=request.retrieval_setting.top_k
        )
        logger.info("============== SERVICE VECTOR SEARCH RETRIEVAL PROCESS ==============")

        filtered_results = filter_results_by_threshold(results, request.retrieval_setting.score_threshold)
        records = convert_to_retrieval_records(filtered_results)
        
        return RetrievalResponse(records=records)
        
    except Exception as e:
        logger.info(f'Retrieval error: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/product_vector_retrieval", response_model=RetrievalResponse)
async def product_vector_retrieval(request: RetrievalRequest,
                                   _: str = Depends(verify_api_key)):
    if milvus_product_vector_retriever is None:
        logger.info("Lỗi: chưa khởi tạo product vectorstore retrieval")
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        results = milvus_product_vector_retriever.retrieve(
            query=request.query, 
            top_k=request.retrieval_setting.top_k
        )
        logger.info("============== PRODUCT VECTOR SEARCH RETRIEVAL PROCESS ==============")

        filtered_results = filter_results_by_threshold(results, request.retrieval_setting.score_threshold)
        records = convert_to_retrieval_records(filtered_results)
        
        return RetrievalResponse(records=records)
        
    except Exception as e:
        logger.info(f'Retrieval error: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    
@app.post("/sql_retrieval", response_model=SQLRetrievalResponse)
async def sql_retrieval(request: SqlRetrievalRequest):
    try:
        search_result = await elastic_sql_retriever.search(request.query)
        
        if not search_result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả phù hợp")

        results, sql_query = search_result
        return SQLRetrievalResponse(results=results, sql_query=sql_query)
        
    except Exception as e:
        logger.info(f'Retrieval error: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "retrieval_app:app",
        host="0.0.0.0",   # <— phải là 0.0.0.0, không phải 127.0.0.1
        port=8004,
        reload=True
    )