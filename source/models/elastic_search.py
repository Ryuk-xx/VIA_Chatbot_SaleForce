from configs.logging_config import load_config
from configs.prompt import PRODUCT_SQL_GENERATION_PROMPT, PRODUCT_SELECTED_COLUMNS, PRODUCT_SQL_SAMPLES, PRODUCT_TABLE_DESCRIPTION, PRODUCT_COLUMN_INFO, PRODUCT_SQL_DOUBLE_CHECK_GENERATION

from source.utils.llm_invoker import invoke_llm_for_full_response
import asyncio
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI 

config = load_config()
llm = ChatOpenAI(
    model=config['llm']['model_4_1'],
    openai_api_key=config['llm']['openai_api_key'],
    temperature=0.1,
)

class BaseElasticSQLRetriever:
    def __init__(self, es, logger, config):
        self.es = es
        self.logger = logger
        self.config = config  # chứa các thông tin cấu hình riêng của từng bảng (product/service)
        
    async def create_sql(self, query):
        sql_genration_prompt = self.config['SQL_GENERATION_PROMPT'].format(
            question=query,
            table_name=self.config['TABLE_NAME'],
            table_description=self.config['TABLE_DESCRIPTION'],
            columns_info=self.config['COLUMN_INFO'],
            sql_samples=self.config['SQL_SAMPLES'],
            selected_columns=self.config['SELECTED_COLUMNS']
        )
        response = await invoke_llm_for_full_response(llm, [HumanMessage(content=sql_genration_prompt)])
        return response

    async def create_sql_double_check(self, query, previous_sql, sql_error):
        sql_double_check_prompt = self.config['SQL_DOUBLE_CHECK_PROMPT'].format(
            question=query,
            table_name=self.config['TABLE_NAME'],
            table_description=self.config['TABLE_DESCRIPTION'],
            columns_info=self.config['COLUMN_INFO'],
            sql_samples=self.config['SQL_SAMPLES'],
            selected_columns=self.config['SELECTED_COLUMNS'],
            previous_sql=previous_sql,
            sql_error=sql_error
        )
        response = await invoke_llm_for_full_response(llm, [HumanMessage(content=sql_double_check_prompt)])
        return response

    def check_sql(self, sql_query):
        try:
            resp = self.es.sql.query(
                query=sql_query,
                format="json",
                fetch_size=10
            )
            return True, resp
        except Exception as e:
            return False, e

    def paser_resp(self, resp):
        cols = [c['name'] for c in resp['columns']]
        result_lines = ["Các kết quả được lấy ra từ database:\n"]

        for idx, row in enumerate(resp['rows'], 1):
            item = dict(zip(cols, row))
            item = {k: v for k, v in item.items() if v is not None}

            result_lines.append(f"Mục {idx}:")
            for k, v in item.items():
                result_lines.append(f"{k}: {v}")
            result_lines.append("")

        return "\n".join(result_lines)

    async def search(self, query):
        sql_query = await self.create_sql(query)
        status, resp = self.check_sql(sql_query)

        if status:
            return self.paser_resp(resp), sql_query
        else:
            sql_query_v2 = await self.create_sql_double_check(query, sql_query, resp)
            status_v2, resp_v2 = self.check_sql(sql_query_v2)

            if status_v2:
                return self.paser_resp(resp_v2), sql_query_v2
            else:
                return None
            
class ProductElasticSQLRetriever(BaseElasticSQLRetriever):
    def __init__(self, es, logger):
        product_config = {
            "TABLE_NAME": "products",
            "TABLE_DESCRIPTION": PRODUCT_TABLE_DESCRIPTION,
            "COLUMN_INFO": PRODUCT_COLUMN_INFO,
            "SQL_SAMPLES": PRODUCT_SQL_SAMPLES,
            "SELECTED_COLUMNS": PRODUCT_SELECTED_COLUMNS,
            "SQL_GENERATION_PROMPT": PRODUCT_SQL_GENERATION_PROMPT,
            "SQL_DOUBLE_CHECK_PROMPT": PRODUCT_SQL_DOUBLE_CHECK_GENERATION
        }
        super().__init__(es, logger, product_config)

# 
if __name__ == "__main__":
    from elasticsearch import Elasticsearch, helpers
    
    es_host = config['elasticsearch']['host']
    es_username = config['elasticsearch']['Username']
    es_password = config['elasticsearch']['Password']
    es = Elasticsearch(es_host, basic_auth=(es_username, es_password), verify_certs=False)  

    elastic_sql_retriever = ProductElasticSQLRetriever(es, None)
    resp, sql_query = asyncio.run(elastic_sql_retriever.search("Tìm bộ lưu trữ năng lượng mặt trời có dung lượng pin lớn nhất"))
    print(sql_query)
    print(len(resp))
    print(resp)
    
    
    
    
    
    
    
    