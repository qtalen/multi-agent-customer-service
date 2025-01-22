from pathlib import Path

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.node_parser import (
    SentenceSplitter
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from dotenv import load_dotenv

load_dotenv("../.env")

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)
Settings.embed_model = embed_model


def get_index(collection_name: str,
              files: list[str]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path="temp/.chroma")

    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    ready = collection.count()
    if ready > 0:
        print("File already loaded")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        print("File not loaded.")
        docs = SimpleDirectoryReader(input_files=files).load_data()
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, embed_model=embed_model,
            transformer=[SentenceSplitter(chunk_size=512, chunk_overlap=20)]
        )

    return index


INDEXES = {
    "SKUS": get_index("skus_docs", ["data/skus_en.txt"]),
    "TERMS": get_index("terms_docs", ["data/terms_en.txt"])
}


async def query_docs(
        index: VectorStoreIndex, query: str,
        similarity_top_k: int = 1
) -> str:
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes = await retriever.aretrieve(query)
    result = ""
    for node in nodes:
        result += node.get_content() + "\n\n"
    return result
