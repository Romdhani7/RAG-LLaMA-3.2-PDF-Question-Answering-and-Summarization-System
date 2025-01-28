import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000"

# Streamlit app
st.title("Data Science Paper Chatbot")

# Upload a PDF
st.header("Upload a Data Science Paper")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file:
    # Send the PDF to the FastAPI backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{FASTAPI_URL}/upload_pdf/", files=files)
    if response.status_code == 200:
        st.success("PDF uploaded and processed successfully!")
    else:
        st.error("Failed to upload PDF.")

# Ask a question
st.header("Ask a Question")
query = st.text_input("Enter your question:")
if query:
    # Send the query to the FastAPI backend
    response = requests.post(f"{FASTAPI_URL}/ask_question/", json={"query": query})
    if response.status_code == 200:
        result = response.json()
        st.subheader("Answer:")
        st.write(result["answer"])
    else:
        st.error("Failed to get a response.")

# Get a summary
if st.button("Get Summary"):
    # Request a summary from the FastAPI backend
    response = requests.post(f"{FASTAPI_URL}/get_summary/", json={"query": query})
    if response.status_code == 200:
        result = response.json()
        st.subheader("Summary:")
        st.write("\n".join(result["summary"]))
    else:
        st.error("Failed to get a summary.")
