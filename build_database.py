import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load your cases
print("Loading dataset...")
# Using the filename you verified in your folder earlier
file_path = "jud-ipl.csv" 

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please rename your CSV to match.")
else:
    # Loading first 1000 cases to ensure your RAM stays healthy during testing
    df = pd.read_csv(file_path).head(1000)

    # FIX: Replace NaN/Empty cells with empty strings to avoid 'float' error
    df['judgement'] = df['judgement'].fillna("") 

    # 2. Split long judicial text into smaller chunks
    # This allows the AI to find the most relevant parts of the law
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents(df['judgement'].astype(str).tolist())

    # 3. Choose an Embedding Model (Text-to-Numbers)
    # This model runs perfectly offline on your MSI Thin 15
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and save the Database locally
    print(f"Processing {len(chunks)} chunks into ChromaDB... this may take a moment.")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./legal_db"
    )

    print("Success! Your legal database is saved in the 'legal_db' folder.")