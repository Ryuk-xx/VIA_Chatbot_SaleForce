import json
from kafka import KafkaConsumer
import psycopg2
from psycopg2.extras import execute_values, execute_batch
from configs.data_config import KafkaConfig
from configs.logging_config import setup_logging
from configs.config import load_config
from source.models.elastic_indexing import Elastic_Indexing
from source.models.vector_indexing import MilvusVectorStore
from source.utils.convert_df_to_document import convert_df_to_document
from elasticsearch import Elasticsearch, helpers
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pandas as pd
from langchain_core.documents import Document
from datetime import datetime

config = load_config()
logger = setup_logging("Elastic_Indexing")

# Set up preprocessor
preprocessor = convert_df_to_document()

# SET UP elastic_handler
llm = ChatOpenAI(
    model=config['llm']['model_4_1'],
    openai_api_key=config['llm']['openai_api_key'],
    temperature=0.1,
)
    
es_host = config['elasticsearch']['host']
es = Elasticsearch(es_host)

elastic_handler = Elastic_Indexing("saleforce_products", es, config["product_fields"], llm, recreate_index=False, logger=logger)



# Set up milvus_handler
milvus_uri = config["milvus"]["milvus_uri"]
milvus_embedding_model = OpenAIEmbeddings(
                        openai_api_key=config["llm"]["openai_api_key"],
                        model=config["milvus"]["embedding_model"],
                        dimensions=config["milvus"]["embedding_dimension"],
                    )

product_milvus_handler = MilvusVectorStore(
    milvus_uri=milvus_uri,
    collection_name=config["milvus"]["product_collection_name"],
    embeddings=milvus_embedding_model,
    dimensions=config["milvus"]["embedding_dimension"],
    openai_key=config["llm"]["openai_api_key"],
    recreate_collection=False,
    logger=logger,
)

service_milvus_handler = MilvusVectorStore(
    milvus_uri=milvus_uri,
    collection_name=config["milvus"]["service_collection_name"],
    embeddings=milvus_embedding_model,
    dimensions=config["milvus"]["embedding_dimension"],
    openai_key=config["llm"]["openai_api_key"],
    recreate_collection=False,
    logger=logger,
)
    
class KafkaConfig:
    BROKER = "10.221.194.133:9092"
    # Thêm topic mới "vcc-sync-menu" cùng với "vcc-sync-product"
    TOPIC_ALL = ["vcc-sync-product", "b2c_sync_service_topic"]
    KAFKA_GROUP = "ai_b2c_product_group"


class ProductSyncHandler:
    def __init__(self):
        self.preprocess = preprocessor
        self.milvus_handler = product_milvus_handler
        self.elastic_handler = elastic_handler
        self.conn = psycopg2.connect(
            host="10.248.243.162",
            port=5432,
            user="ai_chatbot_admin",
            password="Vcc#2024#",
            dbname="ai_services"
        )
        self.conn.autocommit = True

    def sku_exists(self, sku):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM m2_datasets.data_products WHERE sku = %s LIMIT 1", (sku,))
            return cur.fetchone() is not None

    def is_changed(self, sku, new_data):
        """So sánh dữ liệu mới với DB hiện tại"""
        query = "SELECT * FROM m2_datasets.data_products WHERE sku = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (sku,))
            row = cur.fetchone()
            if not row:
                return False

            colnames = [desc[0] for desc in cur.description]
            db_data = dict(zip(colnames, row))

            # Các trường cần so sánh
            fields_to_compare = [
            "id", "name", "price", "thumbnail", "images", "weight", 
            "short_description", "description", "salient_features",  "attributes"
            ]
            for field in fields_to_compare:
                if str(db_data.get(field)) != str(new_data.get(field)):
                    return True
            return False

    def delete_by_sku(self, sku):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM m2_datasets.data_products WHERE sku = %s", (sku,))
        
        self.elastic_handler.delete_by_sku(sku)
        self.milvus_handler.delete_by_id(sku)

    def insert_product_batch(self, batch_data):
        if not batch_data:
            return

        new_records = []
        update_records = []

        for data in batch_data:
            sku = data.get("sku")
            if not sku:
                continue

            if not self.sku_exists(sku):
                new_records.append(data)
            elif self.is_changed(sku, data):
                self.delete_by_sku(sku)
                update_records.append(data)

        merged_data = new_records + update_records

        if not merged_data:
            logger.info("⚠️ Không có dữ liệu mới hoặc cập nhật để insert.")
            return

        logger.info(f"✅ Số lượng dữ liệu mới: {len(new_records)}, cập nhật: {len(update_records)}")

        # PostgreSQL insert
        self._insert_to_postgres(merged_data)

        # Preprocessing
        df = pd.DataFrame(merged_data)
        product_documents = self.preprocess.convert(df, 
                                        html_columns=["description", "salient_features", "short_description"], 
                                        json_columns=["attributes"], 
                                        list_columns=['images', 'category_id'],
                                        drop_columns=['services'])
        
        # Elasticsearch add document
        product_documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in product_documents] # list[dict]
        extra_col_product_documents = self.elastic_handler.create_extra_column_json(product_documents)
        self.elastic_handler.add_documents(extra_col_product_documents)
        
        
        # Milvus add document
        product_documents = [json.dumps(item, ensure_ascii=False) for item in product_documents]        # list[str]
        product_documents = [Document(page_content=text) for text in product_documents]
        self.milvus_handler.add_documents(product_documents)

        

    def _insert_to_postgres(self, batch_data):
        query = """
        INSERT INTO m2_datasets.data_products (
            "id", "name", "sku", "price", "thumbnail", "images", "category_id", "weight", 
            "short_description", "description", "salient_features", "services", "attributes"
        ) VALUES %s
        ON CONFLICT ("sku") DO UPDATE SET 
            "name" = EXCLUDED."name",
            "price" = EXCLUDED."price",
            "thumbnail" = EXCLUDED."thumbnail",
            "images" = EXCLUDED."images",
            "category_id" = EXCLUDED."category_id",
            "weight" = EXCLUDED."weight",
            "short_description" = EXCLUDED."short_description",
            "description" = EXCLUDED."description",
            "salient_features" = EXCLUDED."salient_features",
            "services" = EXCLUDED."services",
            "attributes" = EXCLUDED."attributes";
        """

        values = [
            (
                data.get("id"),
                data.get("name"),
                data.get("sku"),
                data.get("price"),
                data.get("thumbnail"),
                json.dumps(data.get("images", [])),
                json.dumps(data.get("categoryId", [])),
                data.get("weight"),
                data.get("shortDescription"),
                data.get("description"),
                data.get("salientFeatures"),
                json.dumps(data.get("services", [])),
                json.dumps(data.get("attributes", []))
            )
            for data in batch_data
        ]

        with self.conn.cursor() as cur:
            execute_values(cur, query, values)



class ServiceSyncHandler:
    def __init__(self):
        self.preprocess = preprocessor
        self.milvus_handler = service_milvus_handler
        self.conn = psycopg2.connect(
            host="10.248.243.162",
            port=5432,
            user="ai_chatbot_admin",
            password="Vcc#2024#",
            dbname="ai_services"
        )
        self.conn.autocommit = True

    def id_exists(self, id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM m2_datasets.data_services WHERE id = %s LIMIT 1", (id,))
            return cur.fetchone() is not None

    def is_changed(self, id, new_data):
        query = "SELECT * FROM m2_datasets.data_services WHERE id = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (id,))
            row = cur.fetchone()
            if not row:
                return False

            colnames = [desc[0] for desc in cur.description]
            db_data = dict(zip(colnames, row))

            # Chỉ so sánh các trường liên quan
            fields = [
                "id", "created_at", "updated_at", "code", "description", "menu_code", "name", "order",
                "price", "type", "status", "unit", "value_type", "vat"
            ]
            for field in fields:
                if str(db_data.get(field)) != str(new_data.get(field)):
                    return True
            return False

    def delete_by_id(self, id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM m2_datasets.data_services WHERE id = %s", (id,))
        self.milvus_handler.delete_by_id(id)

    def insert_service_batch(self, batch_data):
        if not batch_data:
            return

        new_records = []
        update_records = []

        for data in batch_data:
            id = data.get("id")
            if not id:
                continue

            if not self.id_exists(id):
                new_records.append(data)
            elif self.is_changed(id, data):
                self.delete_by_id(id)
                update_records.append(data)

        merged_data = new_records + update_records

        if not merged_data:
            logger.info("⚠️ Không có dữ liệu mới hoặc cập nhật để insert.")
            return

        logger.info(f"✅ Số lượng dịch vụ mới: {len(new_records)}, cập nhật: {len(update_records)}")

        self._insert_to_postgres(merged_data)

        df = pd.DataFrame(merged_data)
        service_docs = self.preprocess.convert(df, drop_columns=[])

        service_docs = [json.dumps(item, ensure_ascii=False) for item in service_docs]
        service_docs = [Document(page_content=doc) for doc in service_docs]
        self.milvus_handler.add_documents(service_docs)

    def _insert_to_postgres(self, batch_data):
        query = """
        INSERT INTO m2_datasets.data_services (
            id, created_at, updated_at, code, description, menu_code, name, "order",
            price, type, status, unit, value_type, vat
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            created_at = EXCLUDED.created_at,
            updated_at = EXCLUDED.updated_at,
            code = EXCLUDED.code,
            description = EXCLUDED.description,
            menu_code = EXCLUDED.menu_code,
            name = EXCLUDED.name,
            "order" = EXCLUDED."order",
            price = EXCLUDED.price,
            type = EXCLUDED.type,
            status = EXCLUDED.status,
            unit = EXCLUDED.unit,
            value_type = EXCLUDED.value_type,
            vat = EXCLUDED.vat
        """

        values = [
            (
                data.get("id"),
                datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None,
                datetime.fromisoformat(data.get("updated_at")) if data.get("updated_at") else None,
                data.get("code"),
                data.get("description"),
                data.get("menu_code"),
                data.get("name"),
                data.get("order"),
                data.get("price"),
                data.get("type"),
                data.get("status"),
                data.get("unit"),
                data.get("value_type"),
                data.get("vat")
            )
            for data in batch_data
        ]

        with self.conn.cursor() as cur:
            execute_values(cur, query, values)






def process_kafka_messages():
    """
    Lắng nghe Kafka topics và xử lý từng loại dữ liệu:
    - Product → ProductSyncHandler
    - Service → ServiceSyncHandler
    """
    consumer = KafkaConsumer(
        *KafkaConfig.TOPIC_ALL,  # bao gồm cả "vcc-sync-product", "vcc-sync-service"
        bootstrap_servers=[KafkaConfig.BROKER],
        group_id=KafkaConfig.KAFKA_GROUP,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    product_handler = ProductSyncHandler()
    service_handler = ServiceSyncHandler()

    batch_products = []
    batch_services = []

    batch_size = 10  # tùy chỉnh

    try:
        logger.info("🔄 Bắt đầu lắng nghe Kafka messages ...")
        for message in consumer:
            topic = message.topic
            data = message.value

            logger.info(f"📩 Nhận message từ topic [{topic}]: {data}")

            if not data:
                logger.info(f"⚠️ [{topic}] Message rỗng, bỏ qua...")
                continue

            # Parse dữ liệu
            if isinstance(data, dict):
                raw_data = data.get("data", data)
            elif isinstance(data, list):
                raw_data = data
            else:
                logger.info(f"[{topic}] Dữ liệu không hợp lệ, bỏ qua...")
                continue

            if isinstance(raw_data, str):
                try:
                    raw_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ [{topic}] Dữ liệu không hợp lệ JSON, bỏ qua...")
                    continue

            data_list = raw_data if isinstance(raw_data, list) else [raw_data]

            # Phân loại xử lý theo topic
            if topic == "vcc-sync-product":
                batch_products.extend(data_list)
                if len(batch_products) >= batch_size:
                    product_handler.insert_product_batch(batch_products)
                    batch_products = []

            elif topic == "b2c_sync_service_topic":
                batch_services.extend(data_list)
                if len(batch_services) >= batch_size:
                    service_handler.insert_service_batch(batch_services)
                    batch_services = []

    except KeyboardInterrupt:
        logger.info("🛑 Dừng listener Kafka...")
    finally:
        # Flush dữ liệu còn lại
        if batch_products:
            product_handler.insert_product_batch(batch_products)
        if batch_services:
            service_handler.insert_service_batch(batch_services)
        consumer.close()

if __name__ == "__main__":
    process_kafka_messages()
