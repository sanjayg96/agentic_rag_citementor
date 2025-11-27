# ⚖️ CiteMentor | The Attribution-Aware Knowledge Engine

### *Pay only for the wisdom you use.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)

## Executive Overview

**CiteMentor** is a specialized GenAI platform that transforms static non-fiction libraries into interactive, actionable mentorship. Unlike generic Large Language Models (LLMs) that provide broad, opaque answers, this system utilizes a curated knowledge base of trusted authors to provide grounded advice.

**The Differentiator:** This project introduces a novel economic framework for AI: a **Micro-Royalty System**. It tracks exactly which content is used to generate an answer (down to the paragraph), paving the way for fair author compensation in the age of AI.

---

## The Problem Statement

Current information consumption models are broken for both consumers and creators:

1.  **The "Implementation Gap" (Readers):** People seek advice from books but struggle with volume. Attention spans are shortening; users want solutions, not just 300 pages of text.
2.  **The "Hallucination Risk" (Generic AI):** Public LLMs often generate convincing but generic or factually incorrect advice. They lack a verifiable "source of truth."
3.  **The "Black Box" Issue (Authors):** Intellectual property is currently used to train models without consent or compensation. There is no clear mechanism to attribute a specific AI answer to a specific author's work.

## The Solution

An AI-assisted "Mentor" that sits on top of a curated library of high-quality non-fiction.

* **Context-Aware Advisory:** Users ask life/career questions (e.g., *"How do I handle a toxic boss?"*). The system retrieves specific strategies from verified experts.
* **Verifiable Trust:** Every answer is accompanied by a **"Source Card"** displaying the exact book, author, and snippet used.
* **The "Fair-Use" Ledger:** A built-in accounting mechanism that calculates cost per query based on the specific chunks retrieved.

| Feature | Benefit to User | Benefit to Ecosystem |
| :--- | :--- | :--- |
| **Curated RAG** | High-quality, specific advice. No hallucinations. | Eliminates "noise" from internet-scraped data. |
| **Citation Engine** | Trust and transparency. You see the evidence. | Drives discovery of original books. |
| **Royalty Logic** | Users pay fractions of a cent per answer. | Solves the ethical dilemma of AI vs. Copyright. |

---

## Technical Architecture

This project moves beyond basic "Chat with PDF" tutorials by implementing an **Agentic RAG** pipeline with advanced retrieval techniques.

![alt text](archetecture_diagram.png)

### 1. Data Ingestion & Contextual Embeddings (Offline Phase)
* **Processing:** PDF documents are processed using **Google Colab (T4 GPU)**.
* **Contextualization:** (Thanks to Anthropic) Instead of simple chunking, I used **`stabilityai/stablelm-zephyr-3b`** to generate a 1-sentence summary of every chunk *and its neighbors*. This summary is pre-pended to the text before embedding.
    * *Why?* A chunk saying "He bought it" is useless to a vector search. A summary saying "The author decides to buy the stock" makes the chunk retrievable.
* **Cost Calculation:** A metadata field calculates the fractional cost of every chunk relative to the book's total price.

### 2. The Retrieval Engine (Hybrid Search)
* **Vector Store:** **ChromaDB** (Separated into collections by Genre).
* **Lexical Search:** **BM25** (Okapi) for keyword precision.
* **Hybrid Logic:** The system retrieves the top K results from both Vector and Keyword searches, de-duplicates them, and passes them to the re-ranker.

### 3. The "Brain" (Inference Phase)
* **Router Agent:** A lightweight LLM (`gpt-4o-mini`) classifies the user query into a specific domain (Finance, Relationships, Philosophy) to query the correct database silo.
* **Re-Ranker:** A simulated cross-encoder evaluates the retrieved chunks and selects only the top 3 most relevant to the query to reduce noise and cost.
* **Response Generation:** The final answer is synthesized using only the selected excerpts, with strict instructions to cite sources implicitly.

### 4. Tech Stack
* **Framework:** LangChain
* **UI:** Streamlit (with custom CSS for Dark Mode & Card UI)
* **Vector DB:** ChromaDB
* **LLMs:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) & StableLM (Local)

---

## Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/CiteMentor-demo.git](https://github.com/yourusername/CiteMentor-demo.git)
    cd CiteMentor-demo
    ```

2.  **Create a Virtual Environment**
    ```bash
    conda create -n CiteMentor python=3.10
    conda activate CiteMentor
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a `.env` file in the root directory:
    ```bash
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxx
    ```

5.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## Project Structure

```text
├── app.py                 # Main Streamlit UI application
├── rag_engine.py          # Core RAG logic (Routing, Hybrid Search, Reranking)
├── ingest.py              # Script to ingest JSON data into ChromaDB
├── processed_books.json   # Pre-processed data (output from Colab)
├── chroma_db/             # Persistent Vector Database
├── bm25_indices.pkl       # Serialized BM25 indices
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```
## Note

For demo purposes, I have used only 3 public domain books.