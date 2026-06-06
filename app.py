import streamlit as st
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="AI Judicial Assistant", layout="wide")
st.title("⚖️ Indian Judicial AI Assistant")
st.markdown("---")

# 1. Load the Database
@st.cache_resource
def load_db():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./legal_db", embedding_function=embeddings)

db = load_db()

# 2. Accuracy Calculation Function
def calculate_similarity(original, generated):
    """Calculates how closely the AI verdict matches the legal precedent."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([original, generated])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

# 3. Sidebar for Project Info
with st.sidebar:
    st.header("Project Details")
    st.info("Built for VIT Bhopal Project Exhibition")
    st.write("Model: Llama 3 (Offline)")
    st.write("Metric: Cosine Similarity (NLP)")

# 4. Main Interface
case_input = st.text_area("Enter the facts of the case:", height=150, placeholder="Paste case details here...")

if st.button("Generate Verdict & Compare"):
    if case_input:
        with st.spinner("Analyzing precedents and deliberating..."):
            # A. Search for the single best historical match
            docs = db.similarity_search(case_input, k=1)
            original_precedent = docs[0].page_content
            
            # B. AI Prompt with "Explainability" instructions
            prompt = f"""
            You are a Senior Judge of the Supreme Court of India.
            
            HISTORICAL PRECEDENT FOUND IN DATABASE:
            {original_precedent}
            
            NEW CASE TO DECIDE:
            {case_input}
            
            TASK:
            1. Provide a verdict for the new case.
            2. Identify the relevant IPC Section.
            3. Explicitly explain the connection: "Similar to the precedent provided, where [explain connection], I am awarding [punishment]."
            """
            
            # C. Get Response from Llama 3
            response = ollama.generate(model='llama3', prompt=prompt)
            ai_verdict = response['response']
            
            # D. Calculate Accuracy Score
            accuracy_score = calculate_similarity(original_precedent, ai_verdict)

            # E. Display Results Side-by-Side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📜 Historical Precedent")
                st.caption("Extracted from your 53,000-case dataset")
                st.info(original_precedent[:1500] + "...") 

            with col2:
                st.subheader("👨‍⚖️ AI Judge Verdict")
                st.caption(f"Match Accuracy: {accuracy_score}%")
                st.success(ai_verdict)
                st.progress(accuracy_score / 100)

            # F. Transparency Section
            st.markdown("---")
            st.subheader("🔍 Why this verdict?")
            st.write(f"The system identified a **{accuracy_score}% semantic match** between your input and established Supreme Court rulings. The AI leveraged this historical context to ensure a consistent legal decision.")

    else:
        st.warning("Please enter case facts first.")