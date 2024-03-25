from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
        self.db = None

    def chunk_data(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150
        )
        docs = text_splitter.create_documents([documents])
        return docs

    def create_embeddings(self,text,save_embeddings = False):
        docs = self.chunk_data(text)
        db = FAISS.from_documents(docs, self.embeddings)
        if save_embeddings:
            db.save_local(folder_path = 'emb')
        self.db = db
    
    def query_search(self,db,prompt):
        similar_doc = db.similarity_search(prompt, k=1)
        context = similar_doc[0].page_content
        return context
if __name__ == '__main__':
    q = Query()
    with open('stats.txt','r') as f:
        text = f.read()
    q.create_embeddings(text)
    # print(db)
    question = "What is probabilty?"
    searchDocs = q.query_search(q.db,question)
    print(searchDocs)
# print(searchDocs[0].page_content)
# retriever = searchDocs.as_retriever()
