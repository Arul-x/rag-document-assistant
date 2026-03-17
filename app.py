
import streamlit as st
from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("AI Document Assistant (RAG)")

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# File upload
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf","docx"])

text = ""

# ---------- Extract Text ----------
if uploaded_file:

    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    else:
        doc = docx.Document(uploaded_file)

        for para in doc.paragraphs:
            text += para.text + "\n"

# ---------- After Upload ----------
if text:

    st.success("File uploaded successfully!")

    # -------- Summary --------
    summary_input = text[:2000]

    summary = summarizer(
        summary_input,
        max_length=120,
        min_length=40,
        do_sample=False
    )

    st.subheader("Document Summary")

    st.write(summary[0]['summary_text'])

    # -------- Chunk Text --------
    chunk_size = 500
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    # -------- Embeddings --------
    embeddings = embedding_model.encode(chunks)

    # -------- Vector DB --------
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    # -------- Question Box --------
    question = st.text_input("Ask question about document")

    if question:

        query_embedding = embedding_model.encode([question])

        D, I = index.search(np.array(query_embedding), k=3)

        retrieved_text = ""

        for i in I[0]:
            retrieved_text += chunks[i] + " "

        st.subheader("Answer")

        st.write(retrieved_text[:500])
