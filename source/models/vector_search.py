hybrid_search_params = [
    {
        "anns_field": "dense",    
        "metric_type": "COSINE",   
        "params": {}              
    },
    {
        "anns_field": "sparse",    
        "metric_type": "BM25",     
        "params": {}               
    }
]
class MilvusVectorRetriever:
    def __init__(self, vectorstore, logger):
        self.vectorstore = vectorstore        
        self.hybrid_search_params = hybrid_search_params
        self.logger = logger 
        
    def retrieve(self, query, top_k=5):
        try:
            result = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k,
                param=self.hybrid_search_params,
            )
            return result
        except Exception as e:
            self.logger.info(f"Error retrieving documents from vector store: {e}")
            return []
        
# if __name__ == "__main__":
#     from source.models.vector_indexing import MilvusVectorStore
#     from configs.config import load_config
#     from configs.logging_config import setup_logging
#     from langchain_openai import OpenAIEmbeddings
    
#     logger = setup_logging("Chatbot_SaleForce")
    
#     config = load_config()
#     milvus_uri = config["milvus"]["milvus_uri"]
#     milvus_embedding_dimension = config["milvus"]["embedding_dimension"]
#     embedding_model_name = config["milvus"]["embedding_model"]
    
#     openai_key = config["llm"]["openai_api_key"]
    
#     milvus_embedding_model = OpenAIEmbeddings(
#                             openai_api_key=openai_key,
#                             model=embedding_model_name,
#                             dimensions=milvus_embedding_dimension,
#                         )
   
#     product_vectorstore = MilvusVectorStore(
#         milvus_uri=milvus_uri,
#         collection_name=config["milvus"]["product_collection_name"],
#         embeddings=milvus_embedding_model,
#         dimensions=milvus_embedding_dimension,
#         openai_key=openai_key,
#         recreate_collection=False,
#         logger=logger,
#     ).vectorstore
    
#     vector_search = MilvusVectorRetriever(vectorstore=product_vectorstore, logger=logger)
#     print(vector_search.retrieve("điều hòa", top_k=5))