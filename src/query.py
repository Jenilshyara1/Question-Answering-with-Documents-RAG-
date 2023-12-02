from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class Query:
    def __init__(self) -> None:
        self.modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        self.encode_kwargs = {"normalize_embeddings": False}
        self.model_kwargs = {'device':'cpu'}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.modelPath,  # Provide the pre-trained model's path
            model_kwargs=self.model_kwargs,  # Pass the model configuration options
            encode_kwargs=self.encode_kwargs,  # Pass the encoding options
        )
    def load_dataset(self, uploaded_file):
        documents = [uploaded_file.read().decode()]
        return documents

    def chunk_data(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150
        )
        docs = text_splitter.create_documents(documents)
        return docs

    def create_embeddings(self,uploaded_file,save_embeddings = False):
        documents = self.load_dataset(uploaded_file)
        docs = self.chunk_data(documents)
        db = FAISS.from_documents(docs, self.embeddings)
        if save_embeddings:
            db.save_local(folder_path = 'emb')
        return db

if __name__ == '__main__':
    q = Query()
    db = q.create_embeddings('stats.txt')
# question = "What is cheesemaking?"
# searchDocs = db.similarity_search(question)
# print(searchDocs[0].page_content)
# retriever = searchDocs.as_retriever()
