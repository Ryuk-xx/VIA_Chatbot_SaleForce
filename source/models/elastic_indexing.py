from configs.prompt import CREATE_EXTRA_COLUMN_PROMPT
from langchain_core.messages import HumanMessage
from source.utils.llm_invoker import invoke_llm_for_full_response
from langchain_openai import ChatOpenAI 
from elasticsearch import Elasticsearch, helpers
from configs.config import load_config
from configs.logging_config import setup_logging
import json
import asyncio


class Elastic_Indexing:
    def __init__(self, index_name: str, es: Elasticsearch, fields: list, llm: ChatOpenAI,recreate_index: bool = False, logger = None):
        self.create_extra_column_prompt = CREATE_EXTRA_COLUMN_PROMPT
        self.recreate_index = recreate_index
        self.index_name = index_name
        self.es = es
        self.fields = fields
        self.logger = logger    
        self.llm = llm
        
    def create_extra_column_json(self, products) -> list:
        """
        Từ 1 list các file json thuộc tính của products, trích xuất ra các thuộc tính chung nhất (chỉ sử dụng cho products)
        Input:
        - list_of_json: list các file json thuộc tính
        - llm: ChatOpenAI model để gọi LLM
        Output:
        - list các json thuộc tính chung nhất
        """
        extra_col_products = []
        for product in products:
            product_json = json.dumps(product, ensure_ascii=False, indent=2)

            prompt_input = {"json_data": product_json}
            create_extra_column_prompt = CREATE_EXTRA_COLUMN_PROMPT.format(**prompt_input)

            response = asyncio.run(invoke_llm_for_full_response(self.llm, [HumanMessage(content=create_extra_column_prompt)]))
            try:
                response = json.loads(response)
                merged = {**product, **response}
            except:
                product_sku = product['sku']
                self.logger.info(f"Không thể parse JSON ở sản phẩm: {product_sku}")
                merged = product.copy()
                
            extra_col_products.append(merged)
        return extra_col_products
    

    def create_index(self) -> None:
        properties = {}
        for field in self.fields:
            name = field["name"]
            type_ = field["type"]
            prop = {"type": type_}

            if type_ == "date" and "format" in field:
                prop["format"] = field["format"]

            properties[name] = prop
            
        mapping = {
            "mappings": {
                "properties": properties
            }
        }
        if self.es.indices.exists(index=self.index_name) and self.recreate_index:
            self.es.indices.delete(index=self.index_name)
            
        self.es.indices.create(index=self.index_name, body=mapping)
        self.logger.info(f"Đã tạo index {self.index_name}")
        
    def add_documents(self, products: list) -> None:
        """
        Thêm documents vào index Elasticsearch.
        """
        for product in products:
            for key, value in product.items():
                if type(value) == dict:
                    product[key] = json.dumps(value, ensure_ascii=False)
        
        actions = [
        {
            "_index": self.index_name,
            "_source": product if isinstance(product, dict) else product.to_dict(),
        }
            for product in products
        ]
    
        
        success, errors = helpers.bulk(
            self.es,
            actions,
            stats_only=False,     # nếu True thì sẽ không trả về lỗi
            raise_on_error=False, # không dừng nếu có lỗi
            raise_on_exception=False
        )

        failed_items = [e for e in errors if "index" in e and e["index"].get("status", 200) >= 300]

        self.logger.info(f"Đã index thành công {success} documents vào index {self.index_name}.")
        if len(failed_items) > 0:
            self.logger.info(f"Có {len(failed_items)} documents bị lỗi.")
            for err in failed_items:  
                error_detail = err["index"].get("error", {})
                self.logger.error(f"Lỗi: {error_detail}")
  
    
    
if __name__ == "__main__":
    #setup
    config = load_config()
    logger = setup_logging("Elastic_Indexing")
    
    # llm
    llm = ChatOpenAI(
        model=config['llm']['model_4_1'],
        openai_api_key=config['llm']['openai_api_key'],
        temperature=0.1,
    )
    
    #es
    es_host = config['elasticsearch']['host']
    es_username = config['elasticsearch']['Username']
    es_password = config['elasticsearch']['Password']
    es = Elasticsearch(es_host, basic_auth=(es_username, es_password), verify_certs=False)  

    # products
    with open("data/products.json", "r", encoding="utf-8") as f:
        product_documents = json.load(f)   
    
    product_indexer = Elastic_Indexing("products", es, config["product_fields"], llm, recreate_index=False, logger=logger)
    product_indexer.create_index()
    extra_col_product_documents = product_indexer.create_extra_column_json(product_documents)
    product_indexer.add_documents(extra_col_product_documents)
    

    
    # services
    with open("data/services.json", "r", encoding="utf-8") as f:
        service_documents = json.load(f)
        
    service_indexer = Elastic_Indexing("services", es, config["service_fields"], llm, recreate_index=False, logger=logger)
    service_indexer.create_index()
    service_indexer.add_documents(service_documents)