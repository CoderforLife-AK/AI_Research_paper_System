import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

st.set_page_config(page_title="AI Research Paper Intelligence System", layout="wide")

@st.cache_resource
def load_models_and_data():
    # Load localized dataset and vector index files
    df = pd.read_pickle("data/processed_papers.pkl")
    index = faiss.read_index("models/faiss_index.index")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Using standard text summarization pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    kw_model = KeyBERT()
    
    return df, index, embed_model, summarizer, kw_model

try:
    df, index, embed_model, summarizer, kw_model = load_models_and_data()
except FileNotFoundError:
    st.error("🚨 Vector index files missing. Please open your VS Code terminal and run: `python ingest.py` first.")
    st.stop()

# --- UI Layout ---
st.title("📚 AI Research Paper Intelligence System")
st.markdown("Discover and analyze 50k+ Machine Learning papers instantly via semantic context match.")

query = st.text_input("🔍 Enter your research query:", placeholder="e.g., Using reinforcement learning to optimize autonomous drone navigation")
top_k = st.slider("Select retrieval depth (Top-K):", min_value=1, max_value=5, value=3)

if query:
    with st.spinner("Analyzing semantics & retrieving papers..."):
        # Match query structure to processing schema
        query_vector = embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Pull vector matches
        distances, indices = index.search(query_vector, top_k)
        
        st.subheader(f"✨ Found Top {top_k} Matches")
        
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
            paper = df.iloc[idx]
            
            with st.container():
                st.markdown(f"### {rank+1}. {paper['title']}")
                st.caption(f"📊 **Semantic Relevance Score:** {score:.4f}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📄 Original Abstract Excerpt:**")
                    st.info(paper['abstract'])
                
                with col2:
                    st.markdown("**🤖 AI-Generated Summary:**")
                    
                    # Ensure abstract length is within safe window bounds for DistilBART
                    safe_abstract = paper['abstract'][:1024]
                    try:
                        summary_output = summarizer(safe_abstract, max_length=75, min_length=25, do_sample=False)
                        st.success(summary_output[0]['summary_text'])
                    except Exception:
                        st.warning("⚠️ High sequence length or format exception occurred processing this summary.")
                    
                    st.markdown("**🔑 Extracted Research Keywords:**")
                    keywords = kw_model.extract_keywords(paper['abstract'], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                    kw_display = " ".join([f"`{kw[0]}`" for kw in keywords])
                    st.write(kw_display)
                    
            st.markdown("---")