import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Zepto Platform", layout="wide")
st.title("🛒 Zepto Integrated Data & AI Platform")

tab1, tab2, tab3 = st.tabs(["📊 Module 1: Catalog Data", "🤖 Module 2: Survival Predictor", "💬 Module 3: Support Assistant"])

# ---------------------------------------------------------
# TAB 1: DATA PIPELINE & ANALYTICS
# ---------------------------------------------------------
with tab1:
    st.header("Scraped Book Catalog Analytics")
    db_path = "data_pipeline/zepto_catalog.db"
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df_books = pd.read_sql_query("""
            SELECT b.title, c.category_name, b.price_gbp, b.price_inr, b.rating 
            FROM books b 
            JOIN categories c ON b.category_id = c.category_id
        """, conn)
        conn.close()

        st.metric("Total Books In Catalog", len(df_books))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Rated Books (5 Stars)")
            st.dataframe(df_books[df_books["rating"] == 5].head(10), width="stretch")
        with col2:
            st.subheader("Price Distribution (INR)")
            st.bar_chart(df_books["price_inr"].head(20))
    else:
        st.error("Database not found! Run 'python data_pipeline/database.py' first.")

# ---------------------------------------------------------
# TAB 2: MACHINE LEARNING PREDICTION
# ---------------------------------------------------------
with tab2:
    st.header("Titanic Survival Prediction Engine")
    model_path = "analytics/zepto_titanic_pipeline.joblib"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        st.subheader("Enter Passenger Details:")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 1, 80, 25)
            
        with col_b:
            sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
            parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
            
        with col_c:
            fare = st.number_input("Ticket Fare ($)", 0.0, 500.0, 32.2)
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

        if st.button("Predict Survival"):
            input_df = pd.DataFrame([{
                'pclass': pclass, 'sex': sex, 'age': age,
                'sibsp': sibsp, 'parch': parch, 'fare': fare,
                'embarked': embarked
            }])
            
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            if prediction == 1:
                st.success(f"Result: **Survived** (Confidence: {probability*100:.1f}%)")
            else:
                st.error(f"Result: **Did Not Survive** (Confidence: {(1-probability)*100:.1f}%)")
    else:
        st.error("Model file not found! Run 'python analytics/modeling.py' first.")

# ---------------------------------------------------------
# TAB 3: GENAI RAG ASSISTANT
# ---------------------------------------------------------
with tab3:
    st.header("RAG Customer Support Search")
    faq_path = "support_assistant/docs/faq.txt"
    
    if os.path.exists(faq_path):
        with open(faq_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = [line.strip() for line in text.split("\n") if line.strip()]

        @st.cache_resource
        def load_rag_engine(doc_chunks):
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(doc_chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype("float32"))
            return embedder, index

        embedder, index = load_rag_engine(chunks)
        user_query = st.text_input("Ask a question about Zepto services:", "How long does delivery take?")

        if st.button("Search Knowledge Base"):
            query_emb = embedder.encode([user_query])
            distances, indices = index.search(np.array(query_emb).astype("float32"), k=1)
            
            best_chunk = chunks[indices[0][0]]
            score = distances[0][0]
            
            st.info(f"**Retrieved Answer:** {best_chunk}")
            st.caption(f"L2 Distance Metric: {score:.4f}")
    else:
        st.error("FAQ knowledge base not found at support_assistant/docs/faq.txt!")