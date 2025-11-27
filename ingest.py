import json
import os
import pickle
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

INPUT_FILE = "./data/processed_books_data.json"
CHROMA_DB_DIR = "./chroma_db"
BM25_INDEX_FILE = "bm25_indices.pkl"

def get_safe_collection_name(genre_string):
    """
    Converts genre string to a safe ChromaDB collection name.
    e.g. "Personal Finance" -> "personal_finance"
    """
    return genre_string.lower().replace(" ", "_").replace("/", "_")

def main():
    print("--- STARTING MULTI-COLLECTION INGESTION ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Group Data by Genre
    # Structure: { "Personal Finance": [item1, item2], ... }
    grouped_data = {}
    for item in data:
        genre = item['metadata']['genre']
        if genre not in grouped_data:
            grouped_data[genre] = []
        grouped_data[genre].append(item)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # We will store BM25 indices in a dictionary: { "personal_finance": object, ... }
    bm25_store = {}

    # 2. Process Each Genre Separately
    for genre, items in grouped_data.items():
        collection_name = get_safe_collection_name(genre)
        print(f"\nProcessing Genre: '{genre}' -> Collection: '{collection_name}'")
        
        documents = []
        bm25_corpus = []

        for item in items:
            # LangChain Document
            doc = Document(
                page_content=item['contextualized_text'],
                metadata={
                    "id": item['id'],
                    "original_text": item['original_text'],
                    "title": item['metadata']['title'],
                    "author": item['metadata']['author'],
                    "genre": item['metadata']['genre'],
                    "cost": item['metadata']['cost']
                }
            )
            documents.append(doc)
            
            # BM25 Tokenization
            tokens = word_tokenize(item['contextualized_text'].lower())
            bm25_corpus.append(tokens)

        # A. Create Vector Collection for this Genre
        print(f"  - Ingesting {len(documents)} chunks into ChromaDB...")
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR,
            collection_name=collection_name 
        )

        # B. Create BM25 Index for this Genre
        print(f"  - Building BM25 index...")
        bm25_obj = BM25Okapi(bm25_corpus)
        
        # Store both the index and the docs (to map back later)
        bm25_store[collection_name] = {
            "index": bm25_obj,
            "documents": documents
        }

    # 3. Save the Multi-Genre BM25 Store
    print(f"\nSaving separate BM25 indices to {BM25_INDEX_FILE}...")
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_store, f)
    
    print("\n--- INGESTION COMPLETE ---")
    print(f"Collections created: {list(bm25_store.keys())}")

if __name__ == "__main__":
    main()