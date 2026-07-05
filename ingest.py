import os
import pandas as pd
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    print("📥 Loading CShorten/ML-ArXiv-Papers from Hugging Face...")
    # This dataset only has a 'train' split which loads the underlying CSV
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # Let's inspect columns silently, handling potential formatting issues
    print(f"📊 Dataset loaded successfully. Total papers available: {len(df)}")
    
    # Let's clean the columns and handle names (expected: 'title', 'abstract')
    df['title'] = df['title'].fillna('').astype(str).str.strip()
    df['abstract'] = df['abstract'].fillna('').astype(str).str.strip()
    
    # --- OPTIMIZATION STEP FOR LOCAL VS CODE DEVELOPMENT ---
    # The full dataset contains 117k+ rows. Let's slice it to 50,000 papers as planned.
    TARGET_SIZE = 50000 
    if len(df) > TARGET_SIZE:
        print(f"✂️ Slicing dataset down to your target of {TARGET_SIZE} papers for optimized performance.")
        df = df.iloc[:TARGET_SIZE].reset_index(drop=True)
    
    # Combine title and abstract for comprehensive semantic matching
    df['text_to_embed'] = "Title: " + df['title'] + " \nAbstract: " + df['abstract']
    
    print("🧠 Initializing Sentence Transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("⚡ Encoding text into 384-dimensional vectors (This builds your search memory)...")
    # Using a steady batch size to protect CPU memory limits
    embeddings = model.encode(
        df['text_to_embed'].tolist(), 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    print("🧱 Creating FAISS Index (Using Inner Product / Cosine Similarity)...")
    dimension = embeddings.shape[1] # 384 dimensions
    index = faiss.IndexFlatIP(dimension) 
    
    # Convert and normalize vectors for precise cosine similarity calculations
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print("💾 Saving structural checkpoints to disk...")
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save processed DataFrame and Index
    df.to_pickle("data/processed_papers.pkl")
    faiss.write_index(index, "models/faiss_index.index")
    
    print("\n🎉 Pipeline established successfully!")
    print(f"Stored {len(df)} papers inside 'data/processed_papers.pkl'")
    print("Vector database index built inside 'models/faiss_index.index'")

if __name__ == "__main__":
    main()