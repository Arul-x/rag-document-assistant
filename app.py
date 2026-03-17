import streamlit as st
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("AI Document Assistant (RAG)")

# Load models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = load_embedding_model()

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return tokenizer, model

tokenizer, model = load_summarizer()


def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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
