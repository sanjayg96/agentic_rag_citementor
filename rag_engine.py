import sys
import os

# --- CHROMA DB FIX FOR STREAMLIT CLOUD ---
# This ensures the app uses the bundled SQLite binary instead of the old system one.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # If running locally without pysqlite3, this is fine.

import pickle
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Load Environment
load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
BM25_INDEX_FILE = "bm25_indices.pkl"

class RAGEngine:
    def __init__(self):
        print("Initializing RAG Engine...")
        # 1. Load Embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 2. Load BM25 Indices
        if os.path.exists(BM25_INDEX_FILE):
            with open(BM25_INDEX_FILE, "rb") as f:
                self.bm25_store = pickle.load(f)
        else:
            self.bm25_store = {}
            print("WARNING: BM25 Index file not found.")
            
        # 3. Setup LLMs
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.reranker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.generator_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)

    def route_query(self, query: str) -> str:
        """
        Classifies query into one of the known genres or 'unknown'.
        """
        system_prompt = """
        You are a smart library router. Classify the user's query into EXACTLY one of these genres:
        
        1. personal_finance (Keywords: money, debt, savings, wealth, income, expenses, investment)
        2. romantic_relationships (Keywords: love, dating, marriage, partners, spouse, feelings, sex)
        3. mindset_philosophy (Keywords: thoughts, mind, destiny, character, soul, thinking)
        
        If the query clearly doesn't fit any of these, return: "unknown"
        
        RETURN ONLY THE CATEGORY NAME (e.g., "personal_finance"). DO NOT add punctuation or explanation.
        """
        
        messages = [
            ("system", system_prompt),
            ("user", query)
        ]
        
        response = self.router_llm.invoke(messages)
        return response.content.strip().lower()

    def hybrid_search(self, query: str, genre: str, top_k: int = 5) -> List[Document]:
        """
        Performs Vector Search + BM25 Search and combines results.
        """
        # Ensure we have the index
        if genre not in self.bm25_store:
            return []

        # A. Vector Search (Chroma)
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=genre
        )
        # Fetch slightly more for vector search to allow for intersection
        vector_docs = vector_db.similarity_search(query, k=top_k)
        
        # B. BM25 Search (Lexical)
        bm25_data = self.bm25_store[genre]
        bm25_index = bm25_data["index"]
        source_docs = bm25_data["documents"]
        
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = bm25_index.get_scores(tokenized_query)
        
        # Get indices of top scores
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_docs = [source_docs[i] for i in top_indices]
        
        # C. De-duplication (Combine lists)
        seen_ids = set()
        unique_docs = []
        
        # Interleave results: 1 vector, 1 bm25, etc.
        max_len = max(len(vector_docs), len(bm25_docs))
        for i in range(max_len):
            if i < len(vector_docs):
                doc = vector_docs[i]
                doc_id = doc.metadata.get("id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            
            if i < len(bm25_docs):
                doc = bm25_docs[i]
                doc_id = doc.metadata.get("id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
                    
        return unique_docs

    def rerank_chunks(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        """
        Uses LLM to evaluate relevance of each chunk and pick the best ones.
        """
        if not docs:
            return []
            
        # Format candidates for the LLM
        candidates_text = ""
        for i, doc in enumerate(docs):
            # We use the Contextualized Text (Summary+Content) for judging relevance
            candidates_text += f"--- CHUNK {i} ---\n{doc.page_content}\n\n"

        system_prompt = f"""
        You are an expert Re-ranker. 
        User Query: "{query}"
        
        I have provided {len(docs)} text chunks. Your job is to select the Top {top_n} chunks that best answer the query.
        
        Return your answer as a raw JSON list of integers representing the indices of the selected chunks. 
        Example: [0, 4, 1]
        
        Do not explain. Just JSON.
        """
        
        messages = [
            ("system", system_prompt),
            ("user", candidates_text)
        ]
        
        try:
            response = self.reranker_llm.invoke(messages)
            content = response.content.replace("```json", "").replace("```", "").strip()
            selected_indices = json.loads(content)
            
            # Validate indices
            final_docs = [docs[i] for i in selected_indices if i < len(docs)]
            return final_docs
        except Exception as e:
            print(f"Reranking failed: {e}. Returning original top {top_n}.")
            return docs[:top_n]

    def generate_answer(self, query: str, genre: str) -> Dict[str, Any]:
        """
        Main pipeline function: Route -> Search -> Rerank -> Answer
        """
        # 1. Routing (Already done by UI, but good to have safety)
        if not genre:
            genre = self.route_query(query)
            
        if genre == "unknown":
            # Fallback for general questions
            fallback_response = self.generator_llm.invoke(f"User Question: {query}\n\nAnswer this general question politely, but mention that you specialize in Finance, Relationships, and Philosophy books.")
            return {
                "answer": fallback_response.content,
                "sources": [],
                "total_cost": 0.0,
                "genre": "General"
            }

        # 2. Hybrid Search
        raw_docs = self.hybrid_search(query, genre, top_k=6)
        
        # 3. Rerank
        top_docs = self.rerank_chunks(query, raw_docs, top_n=3)
        
        # 4. Synthesize Answer
        context_text = ""
        total_cost = 0.0
        sources_metadata = []

        for doc in top_docs:
            # Accumulate cost
            cost = doc.metadata.get("cost", 0)
            total_cost += cost
            
            # Prepare context for LLM (Use Original Text for answer generation, it flows better)
            original_text = doc.metadata.get("original_text", "")

            # Create a snippet for the UI (first 100 chars)
            # Remove newlines for cleaner UI display
            clean_text = original_text.replace("\n", " ")
            preview = clean_text[:120] + "..." if len(clean_text) > 120 else clean_text

            context_text += f"<excerpt>\n{original_text}\n</excerpt>\n\n"
            
            # Prepare metadata for UI
            sources_metadata.append({
                "book": doc.metadata.get("title"),
                "author": doc.metadata.get("author"),
                "cost": cost,
                "preview": preview
            })

        system_prompt = """
        You are a helpful assistant answering questions based ONLY on the provided book excerpts.
        Cite the excerpts implicitly in your answer. 
        If the excerpts do not contain the answer, say you don't know based on the book.
        """
        
        user_prompt = f"""
        User Question: {query}
        
        Context from Book ({genre}):
        {context_text}
        
        Answer:
        """
        
        response = self.generator_llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        return {
            "answer": response.content,
            "sources": sources_metadata,
            "total_cost": total_cost,
            "genre": genre
        }