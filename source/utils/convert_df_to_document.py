from markdownify import markdownify as md
import re
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime
class convert_df_to_document:
    def normalize_datetime_columns(self, df: pd.DataFrame, datetime_columns: list) -> pd.DataFrame:
        """
        Chuyển định dạng các cột ngày tháng về format: yyyy-MM-dd HH:mm:ss.SSS
        """
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Chuyển đổi, nếu lỗi sẽ là NaT
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]  # Cắt chỉ lấy 3 số mili giây
        return df
    def convert_html_to_markdown(self, html_content: str) -> str:
        """
        Chuyển HTML sang Markdown, tự động:
        """
        if not html_content or isinstance(html_content, float) and pd.isna(html_content):
            return "NULL"
        soup = BeautifulSoup(html_content, "html.parser")

        for style in soup.find_all("style"):
            style.decompose()

        for img in soup.find_all("img"):
            img.decompose()

        for fig in soup.find_all("figure"):
            if not fig.get_text(strip=True):
                fig.decompose()

        for tag in soup.find_all(attrs={"data-pb-style": True}):
            del tag["data-pb-style"]

        clean_html = str(soup)
        markdown_text = md(clean_html, heading_style="ATX", bullets="-")
        markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
        markdown_text = "\n".join(line for line in markdown_text.splitlines() if line.strip())
        return markdown_text or "NULL"

    def convert_list_dict_to_json(self, cell_value):
        """
        Nhận vào 1 ô chứa list các dict. Trả về dict con dạng {code: value, ...}
        """
        if pd.isna(cell_value):
            return {}

        if isinstance(cell_value, str):
            try:
                cell_value = json.loads(cell_value)
            except json.JSONDecodeError:
                return {}

        if isinstance(cell_value, list):
            return {
                item.get("name"): item.get("value")
                for item in cell_value
                if isinstance(item, dict) and "name" in item and "value" in item
            }

        return {}

    def parse_list_string(self, cell_value):
        """
        Xử lý chuỗi dạng list như '["a", "b"]' thành list Python thực sự.
        """
        if pd.isna(cell_value):
            return []

        if isinstance(cell_value, str):
            try:
                parsed = json.loads(cell_value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return []

        if isinstance(cell_value, list):
            return cell_value

        return []

    def convert_df_to_list_json(self, df: pd.DataFrame) -> list:
        def safe_not_null(val):
            if isinstance(val, float):
                return not pd.isna(val)
            return val is not None

        result_series = df.apply(
            lambda row: json.dumps(
                {col: row[col] for col in df.columns if safe_not_null(row[col])},
                ensure_ascii=False
            ),
            axis=1
        )
        return result_series.dropna().tolist()

    def convert(self, df, html_columns=[], json_columns=[], list_columns=[], drop_columns=[], datetime_columns=[]):
        """
        Chuyển đổi DataFrame sang danh sách JSON:
        - html_columns: HTML ➜ Markdown
        - json_columns: list dict ➜ dict con
        - list_columns: chuỗi dạng list ➜ list thực
        """
        clean_df = df.copy()
        clean_df = clean_df.dropna(axis=1, how='all') 
        clean_df = clean_df.drop(columns=drop_columns, errors='ignore') 
        
        for col in html_columns:
            clean_df[col] = clean_df[col].apply(lambda x: self.convert_html_to_markdown(x))

        for col in json_columns:
            clean_df[col] = clean_df[col].apply(lambda x: self.convert_list_dict_to_json(x))

        for col in list_columns:
            clean_df[col] = clean_df[col].apply(lambda x: self.parse_list_string(x))
                
        clean_df = self.normalize_datetime_columns(clean_df, datetime_columns)

        return self.convert_df_to_list_json(clean_df)



if __name__ == "__main__":
        
    import pandas as pd
    import json

    service_path = "data/data_service_29_2_2025.csv"
    product_path = "data/data_products_29_2_2025.csv"
    product_df = pd.read_csv(product_path, encoding="utf-8")
    service_df = pd.read_csv(service_path, encoding="utf-8")


    converter = convert_df_to_document()

    product_documents = converter.convert(product_df, 
                                        html_columns=["description", "salient_features", "short_description"], 
                                        json_columns=["attributes"], 
                                        list_columns=['images', 'category_id'],
                                        drop_columns=['services'])
    service_documents = converter.convert(service_df, 
                                        html_columns=['description'],
                                        datetime_columns=['created_at', 'updated_at'])

    service_documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in service_documents]
    product_documents = [json.loads(doc) if isinstance(doc, str) else doc for doc in product_documents]

    with open("data/products.json", "w", encoding="utf-8") as f:
        json.dump(product_documents, f, ensure_ascii=False, indent=2)
        
    with open("data/services.json", "w", encoding="utf-8") as f:
        json.dump(service_documents, f, ensure_ascii=False, indent=2)
