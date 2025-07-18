from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType
from langchain_openai import OpenAIEmbeddings

class MilvusVectorStore:
    def __init__(self, milvus_uri, collection_name, embeddings, dimensions, openai_key, recreate_collection = False, logger = None):
        self.uri = milvus_uri
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.dimensions = dimensions
        self.openai_key = openai_key
        self.recreate_collection = recreate_collection
        self.logger = logger
        self.vectorstore = self.create_vectorstore()

        
    def create_vectorstore(self):
       
        client = MilvusClient(
            uri=self.uri,
            token="root:Milvus"
        )

        schema = MilvusClient.create_schema(auto_id=False)  # <- Tắt auto_id vì ta muốn tự set sku làm ID

        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
            description="id service hoặc sku product"
        )


        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=32768,
            enable_analyzer=True,
            description="orginal text"
        )

        schema.add_field(
            field_name="dense",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimensions,
            description="Dense embedding"
        )

        schema.add_field(
            field_name="sparse",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
            description="Sparse embedding"
        )

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25
        )
        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )

        if client.has_collection(self.collection_name) and not self.recreate_collection:
            self.logger.info(f"Collection {self.collection_name} already exists. Skipping collection creation.")
        else:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            self.logger.info(f"Collection {self.collection_name} created.")
            
        vector_store = Milvus(
            auto_id=False,
            embedding_function=self.embeddings,
            builtin_function= BM25BuiltInFunction(),
            vector_field = ["dense", "sparse"],
            connection_args={"uri": self.uri,},
            collection_name = self.collection_name
        )
        
        client.close()
        return vector_store
    
    def add_documents(self, documents):
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [doc.metadata["pk"] for doc in documents]  
            self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)
            
            self.logger.info(f"✅ Thêm {len(documents)} documents vào vectorstore: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi thêm documents: {e}")
        
    def delete_by_id(self, id):
        try:
            self.vectorstore.delete(ids=[id])

            self.logger.info(f"✅ Đã xóa document với id = {id} khỏi collection {self.collection_name}")
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi xóa id = {id}: {e}")
                

if __name__ == "__main__":
    import pandas as pd
    from source.utils.convert_df_to_document import convert_df_to_document
    from langchain_core.documents import Document
    from configs.config import load_config
    from configs.logging_config import setup_logging
    
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
       
    import json
    with open("data/products.json", "r", encoding="utf-8") as f:
        raw_product_documents = json.load(f)

    with open("data/services.json", "r", encoding="utf-8") as f:
        raw_service_documents = json.load(f)
        
    product_documents = [
        Document(
            page_content=json.dumps(item, ensure_ascii=False),
            metadata={"pk": item.get("sku", "")}
        )
        for item in raw_product_documents
    ]

    service_documents = [
        Document(
            page_content=json.dumps(item, ensure_ascii=False),
            metadata={"pk": item.get("id", "")}
        )
        for item in raw_service_documents
    ]
    
    product_vectorstore = MilvusVectorStore(
        milvus_uri=milvus_uri,
        collection_name="SaleForce_product_vectorstore",
        embeddings=milvus_embedding_model,
        dimensions=milvus_embedding_dimension,
        openai_key=openai_key,
        recreate_collection=False,
        logger=logger,
    )
    
    service_vectorstore = MilvusVectorStore(       
        milvus_uri=milvus_uri,
        collection_name="SaleForce_service_vectorstore",
        embeddings=milvus_embedding_model,
        dimensions=milvus_embedding_dimension,
        openai_key=openai_key,
        recreate_collection=False,  
        logger=logger,
    )   
    
    # product_vectorstore.delete_by_id("04100100031")
    
    
    
    # def chunk_documents(documents, batch_size=50):
    #     for i in range(0, len(documents), batch_size):
    #         yield documents[i:i+batch_size]

    # for chunk in chunk_documents(product_documents, batch_size=50):
    #     try:
    #         product_vectorstore.add_documents(chunk)
    #     except Exception as e:
    #         logger.error(f"Error adding chunk: {e}")
    
    
    # for chunk in chunk_documents(service_documents, batch_size=50):
    #     try:
    #         service_vectorstore.add_documents(chunk)
    #     except Exception as e:
    #         logger.error(f"Error adding chunk: {e}")
            
        
        
        