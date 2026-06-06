# ⚖️ AI-Powered Indian Judicial Assistant

An advanced, privacy-focused decision-support system designed to automate **Legal Judgement Prediction (LJP)** and provide **Explainable Sentencing**. This application operates **100% offline and locally**, ensuring total data confidentiality for sensitive legal information. 

Instead of functioning as an uninterpretable "black box," the framework utilizes **Retrieval-Augmented Generation (RAG)** to mathematically validate and justify its conclusions using verified historical records.

---

## 🚀 Core Features

* **Semantic Precedent Retrieval:** Accepts raw case facts and executes a sub-second semantic similarity search across a structured database of over **53,000 Indian Supreme Court judgments** (the *Jud-IPL dataset*).
* **Generative Legal Reasoning:** Feeds retrieved precedents and current case facts into a local Large Language Model to draft structural verdicts, determine applicable **IPC Sections**, and suggest appropriate sentencing guidelines.
* **Explainable AI Dashboard:** Features a clean, side-by-side user interface that maps out the logical connection between the retrieved historical precedent and the current case verdict.
* **Algorithmic Validation:** Computes a precise **Semantic Match Accuracy** score using advanced NLP text-vectorization metrics to mathematically back up the AI's predictions.

---

## 🛠️ Tech Stack

* **User Interface:** `Streamlit` (Interactive web-based dashboard and side-by-side layout rendering)
* **Generative Core:** `Ollama` + `Llama 3 (8B)` (Local, offline LLM engine for inference and text generation)
* **Vector Database:** `ChromaDB` (Local vector store handling high-dimensional semantic queries)
* **Embedding Model:** Hugging Face `all-MiniLM-L6-v2` (Sentence Transformer generating 384-dimensional dense text vectors)
* **Data Pipeline & Orchestration:** `LangChain` & `NLTK` (Recursive text splitting into 1,000-character context-preserving overlapping chunks)
* **Statistical Verification:** `Scikit-learn` (`TfidfVectorizer` and `cosine_similarity` matrix calculations for accuracy verification)
* **Environment:** `Python 3` (Isolated virtual workspaces optimized for local GPU acceleration)

---

## 📊 Evaluation & Metrics

* **Retrieval Dataset Size:** 53,000+ Supreme Court Documents (Jud-IPL Corpus)
* **Validation Metric:** Cosine Similarity over TF-IDF Vector Spaces
* **Average Semantic Match Accuracy:** **68.15%** mapping convergence against verified historical precedents

---

## 💻 How to Run Locally

### 1. Prerequisite (Start the LLM Engine)
Ensure Ollama is running locally in your background terminal with the proper architecture pulled:
```powershell
ollama serve
ollama pull llama3
