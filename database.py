import pandas as pd
import chromadb
import uuid

FILE_PATH = "my_portfolio.csv"

class Portfolio:
    def __init__(self, file_path=FILE_PATH):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        if not skills or not isinstance(skills, list):
            return []
            
        try:
            results = self.collection.query(
                query_texts=skills,
                n_results=2
            )
            return results.get('metadatas', [])
        except Exception as e:
            print(f"Error querying links: {str(e)}")
            return []