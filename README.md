📚 **AI Research Paper Intelligence System**
An AI-powered semantic search engine that helps researchers discover, summarize, and analyze 50,000+ ArXiv Machine Learning research papers. Instead of relying on rigid, word-for-word keyword matching, this system uses advanced Natural Language Processing (NLP) to understand the underlying contextual meaning of a user's query.

🚀 **Why This Project?**
Traditional academic search engines often rely on exact keyword matching. For example, if you search for "Deep learning for medical image analysis", a standard engine might completely miss highly relevant papers that discuss computer vision, neural networks, or MRI anomaly classification simply because they used alternative phrasing.

This project completely solves that limitation through Semantic Search, allowing the system to comprehend abstract intent. Furthermore, it implements a Zero-Disk-Storage Cloud Architecture, offloading massive AI model computations to the cloud to run instantaneously on local devices without exhausting hard drive space or RAM.

✨ **Features**
🔍 Context-Aware Semantic Search: Understands technical relationships (e.g., matching LLMs with Generative AI or Transformers).

🧱 Ultrafast Vector DB Indexing: Matches high-dimensional text coordinates locally in milliseconds using FAISS.

🤖 Cloud-Driven Summarization: Generates quick, readable, 2-line summaries of long, dense academic abstracts using a cloud-hosted DistilBART pipeline.

🔑 Intelligent Keyword Extraction: Leverages zero-shot prompting over an open LLM to instantly extract core research themes.

🖥️ Interactive Web Interface: A lightweight, clean Streamlit multi-page experience designed for rapid prototyping.

🏗️ System Architecture & Working
The system is split into two asynchronous execution stages to isolate core processes and minimize hardware reliance:
**====================================== STAGE 1: INGESTION PIPELINE ======================================**
  [ Hugging Face Hub ] ------(Streaming)------> [ pandas DataFrame ] 
  (CShorten/ML-ArXiv-Papers)                      (First 50,000 Entries)
                                                          │
                                                    [ text_to_embed ]
                                            (Title + Abstract Construction)
                                                          │
                                                    (API Request)
                                                          ▼
                                          [ HF Serverless Inference API ]
                                            (all-MiniLM-L6-v2 Embeddings)
                                                          │
                                                   (384-Dim Vectors)
                                                          ▼
  [ models/faiss_index.index ] <───(Save)───── [ FAISS Vector Store ]
  [ data/processed_papers.pkl ] <──(Save)───── [ Saved Clean Text Data ]

**======================================= STAGE 2: STREAMLIT APP =======================================**
  [ User Search Query ] ───> [ HF Inference Engine ] ───> [ Query Vector ] 
                                                                 │
                                                                 ▼
      [ Streamlit Frontend UI ] <──(Render)─── [ Local FAISS Vector Search ] (Top-K Matches)
                  │
                  ├───> [ Cloud DistilBART-CNN ] ──────> [ AI-Generated Abstract Summaries ]
                  └───> [ Cloud Zephyr-7B LLM ] ───────> [ Real-time Keyword Extractions ]

**Detailed Working Mechanics:**
The Ingestion Engine (ingest.py): Streams the dataset over the web. Text inputs are split and sent in secure, compact chunks to the Hugging Face all-MiniLM-L6-v2 API, returning dense vector arrays. The resulting coordinate matrix is converted to a float32 array, L2-normalized, and indexed inside a local FAISS Index Flat Inner Product (IndexFlatIP) matrix to handle rapid cosine similarity calculations.

The Retrieval & Analytics Web Server (app.py): When a user submits a search query, it is vectorized instantly via the cloud. FAISS cross-references the coordinates locally in milliseconds to isolate the Top-K nearest neighbors. The abstracts of these matches are pulled from the disk pickle file and passed to secondary models (DistilBART for textual summarization and a fine-tuned LLM for zero-shot keyword isolation) before rendering the payload directly onto the browser screen.

🛠️ **Tech Stack**
Language: Python

Dataset: Hugging Face CShorten/ML-ArXiv-Papers (110k+ machine learning publications)

Vector Store Matrix: FAISS (Facebook AI Similarity Search)

Embedding Model Node: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional output space)

Summarization Model Node: sshleifer/distilbart-cnn-12-6

Keyword Inference Model Node: HuggingFaceH4/zephyr-7b-beta

Web UI Framework: Streamlit

Core Utilities: Pandas, NumPy, JSON

🌟 **Future Scope**
Hybrid Search Engine: Merging traditional BM25 keyword frequencies with dense semantic structures.

Cross-Encoder Re-ranking: Adding a secondary deep neural net layer to re-sort Top-K results for maximum textual relevance.

Citation Mapping: Visualizing connectivity networks between retrieved publications via Graph networks.

RAG Chat Assistant: Integrating a chat window allowing users to converse directly with retrieved documents.
