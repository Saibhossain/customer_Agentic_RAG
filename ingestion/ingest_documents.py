from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from embeddings.vector_store import VectorStore

def ingest_pdfs(folder):
    vs = VectorStore()
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=100)

    for pdf in folder:
        docs = PyPDFLoader(pdf).load()
        chunks = splitter.split_documents(docs)

        texts = [c.page_content for c in chunks]
        meta = [{
            "type": "financial_doc",
            "source": pdf
        }] * len(texts)

        vs.add(texts, meta)
