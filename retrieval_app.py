from fastapi import FastAPI, Header, HTTPException, Response
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

def verify_api_key(authorization: str = Header(None)) -> bool:
    """Verify the API key from Authorization header"""
    if not authorization:
        return False
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return False
        return len(token) > 0
    except ValueError:
        return False
    
@app.post("/service_vector_retrieval", response_model=RetrievalResponse)
async def retrieval_endpoint(request: RetrievalRequest,
                             authorization: Optional[str] = Header(None)):
    
    # if not verify_api_key(authorization):
    #     raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    # Check if retriever is initialized
    if milvus_service_vector_retriever is None:
        logger.info("Lỗi: chưa khởi tạo service vectorstore retrieval")
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        # Perform retrieval
        results = milvus_service_vector_retriever.retrieve(
            query=request.query, 
            top_k=request.retrieval_setting.top_k
        )
        logger.info("============== VECTOR SEARCH RETRIEVAL PROCESS ==============")
        # Filter results by score threshold
        filtered_results = []
        if request.retrieval_setting.score_threshold:
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= request.retrieval_setting.score_threshold
            ]
        else: 
            filtered_results = [
                (doc, score) for doc, score in results 
            ]
        
        # Convert results to response format
        records = []
        for doc, score in filtered_results:
            content = doc.page_content
            
            parsed_content = json.loads(doc.page_content)
            title = parsed_content.get("name", "Chưa rõ tên")
            
            record = RetrievalRecord(
                content=content,
                score=score,
                title=title,  
                metadata={}  
            )

            records.append(record)
        return RetrievalResponse(records=records)
        
    except Exception as e:
        logger.info(f'Retrieval error: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/retrieval", response_model=RetrievalResponse)
async def retrieval_endpoint(
    request: RetrievalRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Retrieval API endpoint that searches for documents based on query
    """

    # Verify API key
    if not verify_api_key(authorization):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    # Check if retriever is initialized
    if milvus_product_vector_retriever is None:
        logger.info("Lỗi: chưa khởi tạo vector store retrieval")
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        # Perform retrieval
        results = milvus_product_vector_retriever.retrieve(
            query=request.query, 
            top_k=request.retrieval_setting.top_k
        )
        logger.info("============== VECTOR SEARCH RETRIEVAL PROCESS ==============")
        # Filter results by score threshold
        filtered_results = []
        if request.retrieval_setting.score_threshold:
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= request.retrieval_setting.score_threshold
            ]
        else: 
            filtered_results = [
                (doc, score) for doc, score in results 
            ]
        
        # Convert results to response format
        records = []
        for doc, score in filtered_results:
            content = doc.page_content
            
            parsed_content = json.loads(doc.page_content)
            title = parsed_content.get("name", "Chưa rõ tên")
            
            record = RetrievalRecord(
                content=content,
                score=score,
                title=title,  
                metadata={}  
            )

            records.append(record)
        return RetrievalResponse(records=records)
        
    except Exception as e:
        logger.info(f'Retrieval error: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")
    
    

    
@app.post("/sql_retrieval", response_model=SQLRetrievalResponse)
async def retrieval_endpoint(request: SqlRetrievalRequest):
    try:
        search_result = await elastic_sql_retriever.search(request.query)
        
        if not search_result:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả phù hợp")

        results, sql_query = search_result
        return SQLRetrievalResponse(results=results, sql_query=sql_query)
        
    except Exception as e:
        logger.info(f'Retrieval error: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")



# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "vector_store_initialized": retriever is not None
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "retrieval_app:app",
        host="0.0.0.0",   # <— phải là 0.0.0.0, không phải 127.0.0.1
        port=8004,
        reload=True
    )