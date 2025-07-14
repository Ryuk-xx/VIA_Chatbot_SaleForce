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

        schema = MilvusClient.create_schema(auto_id=True)  

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,    
            description="auto id"
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
            auto_id=True,
            embedding_function=self.embeddings,
            builtin_function= BM25BuiltInFunction(),
            vector_field = ["dense", "sparse"],
            connection_args={"uri": self.uri,},
            # index_params= {"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            collection_name = self.collection_name
        )
        
        client.close()
        return vector_store
    
    def add_documents(self, documents):
        try:
            self.vectorstore.add_documents(documents)
            self.logger.info(f"Added {len(documents)} documents to the vector store: {self.collection_name}.")
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
        

    

# if __name__ == "__main__":
#     import pandas as pd
#     from source.utils.convert_df_to_document import convert_df_to_document
#     from langchain_core.documents import Document
#     from configs.config import load_config
#     from configs.logging_config import setup_logging
    
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
   
    
#     service_path = config["path"]["service_path"]
#     product_path = config["path"]["product_path"]
#     product_df = pd.read_csv(product_path, encoding="utf-8")
#     service_df = pd.read_csv(service_path, encoding="utf-8")


#     converter = convert_df_to_document()


#     product_documents = converter.convert(product_df, 
#                                         html_columns=["description", "salient_features", "short_description"], 
#                                         json_columns=["services", "attributes"], 
#                                         list_columns=['images', 'category_id'])
#     service_documents = converter.convert(service_df, 
#                                         html_columns=['description'], 
#                                         json_columns=[], 
#                                         list_columns=[])
    
#     product_documents = [Document(page_content=text) for text in product_documents]
#     service_documents = [Document(page_content=text) for text in service_documents]
    
#     product_vectorstore = MilvusVectorStore(
#         milvus_uri=milvus_uri,
#         collection_name=config["milvus"]["product_collection_name"],
#         embeddings=milvus_embedding_model,
#         dimensions=milvus_embedding_dimension,
#         openai_key=openai_key,
#         recreate_collection=True,
#         logger=logger,
#     )
    
#     service_vectorstore = MilvusVectorStore(       
#         milvus_uri=milvus_uri,
#         collection_name=config["milvus"]["service_collection_name"],
#         embeddings=milvus_embedding_model,
#         dimensions=milvus_embedding_dimension,
#         openai_key=openai_key,
#         recreate_collection=True,  
#         logger=logger,
#     )   
    


#     def chunk_documents(documents, batch_size=100):
#         for i in range(0, len(documents), batch_size):
#             yield documents[i:i+batch_size]

#     for chunk in chunk_documents(product_documents, batch_size=50):
#         try:
#             product_vectorstore.add_documents(chunk)
#         except Exception as e:
#             logger.error(f"Error adding chunk: {e}")
    
    
#     for chunk in chunk_documents(service_documents, batch_size=50):
#         try:
#             service_vectorstore.add_documents(chunk)
#         except Exception as e:
#             logger.error(f"Error adding chunk: {e}")
            
        
        
        