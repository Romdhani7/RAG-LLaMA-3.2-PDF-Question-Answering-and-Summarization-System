# RAG-LLaMA-3.2-PDF-Question-Answering-and-Summarization-System
<h1 align="center">RAG + LLaMA 3.2: PDF QA and Summarization System</h1><p align="center"> <strong>Combining FAISS for semantic search and LLaMA 3.2 for generating responses and summaries.</strong> </p>
<h2>Features</h2><ul> <li><strong>PDF Upload</strong>: Upload and process PDF files.</li> <li><strong>Semantic Search</strong>: Retrieve relevant text chunks using FAISS.</li> <li><strong>Question Answering</strong>: Generate answers using LLaMA 3.2.</li> <li><strong>Summarization</strong>: Provide concise summaries of the content.</li> <li><strong>FastAPI Backend</strong>: RESTful API for easy integration.</li> </ul>
<h2>Prerequisites</h2><p>Before running the project, ensure you have the following installed:</p><ul> <li>Python 3.8+</li> <li><a href="https://github.com/jmorganca/ollama">Ollama</a> (for LLaMA 3.2)</li> <li>FAISS (<code>pip install faiss-cpu</code>)</li> <li>NLTK (<code>pip install nltk</code>)</li> </ul>
<h2>Installation</h2><ol> <li> <strong>Clone the repository:</strong> <pre><code>git clone https://github.com/your-username/your-repo-name.git cd your-repo-name</code></pre> </li> <li> <strong>Install dependencies:</strong> <pre><code>pip install -r requirements.txt</code></pre> </li> <li> <strong>Download NLTK resources:</strong> <pre><code>python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"</code></pre> </li> <li> <strong>Set up Ollama:</strong> <pre><code>ollama pull llama3.2:latest</code></pre> </li> </ol>
<h2>Running the Project</h2><ol> <li> <strong>Start the FastAPI server:</strong> <pre><code>uvicorn main:app --reload</code></pre> </li> <li> <strong>Access the API at:</strong> <a href="http://localhost:8000">http://localhost:8000</a> </li> </ol>
<h2>API Endpoints</h2><h3>1. Upload a PDF</h3> <ul> <li><strong>Endpoint</strong>: <code>POST /upload_pdf/</code></li> <li><strong>Description</strong>: Upload a PDF for processing.</li> <li><strong>Request</strong>: <ul> <li><code>file</code>: PDF file (max 200MB).</li> </ul> </li> <li><strong>Response</strong>: <ul> <li>Success: <code>{"message": "PDF uploaded and processed successfully."}</code></li> <li>Error: HTTP 500 with details.</li> </ul> </li> </ul><h3>2. Ask a Question</h3> <ul> <li><strong>Endpoint</strong>: <code>POST /ask_question/</code></li> <li><strong>Description</strong>: Ask a question based on the uploaded PDF.</li> <li><strong>Request</strong>: <ul> <li>JSON body: <code>{"query": "Your question here"}</code></li> </ul> </li> <li><strong>Response</strong>: <ul> <li>Success: <code>{"summary": "Summary of the context", "answer": "Answer to the question"}</code></li> <li>Error: HTTP 404 or 500 with details.</li> </ul> </li> </ul><h3>3. Get a Summary</h3> <ul> <li><strong>Endpoint</strong>: <code>POST /get_summary/</code></li> <li><strong>Description</strong>: Get a summary of the context relevant to a query.</li> <li><strong>Request</strong>: <ul> <li>JSON body: <code>{"query": "Your query here"}</code></li> </ul> </li> <li><strong>Response</strong>: <ul> <li>Success: <code>{"summary": "Summary of the context"}</code></li> <li>Error: HTTP 404 or 500 with details.</li> </ul> </li> </ul>
<h2>Example Workflow</h2><ol> <li> <strong>Upload a PDF:</strong> <pre><code>curl -X POST -F "file=@example.pdf" http://localhost:8000/upload_pdf/</code></pre> </li> <li> <strong>Ask a question:</strong> <pre><code>curl -X POST -H "Content-Type: application/json" -d '{"query": "What is the main topic?"}' http://localhost:8000/ask_question/</code></pre> </li> <li> <strong>Get a summary:</strong> <pre><code>curl -X POST -H "Content-Type: application/json" -d '{"query": "Summarize the document"}' http://localhost:8000/get_summary/</code></pre> </li> </ol>
<h2>Configuration</h2><ul> <li> <strong>FAISS Index Path</strong>: Update <code>load_faiss_index</code> in <code>main.py</code> to point to your FAISS index file. </li> <li> <strong>Dataset Directory</strong>: Update <code>load_text_chunks_from_directory</code> in <code>main.py</code> to point to your dataset directory. </li> </ul>
<h2>License</h2><p>This project is licensed under the <strong>MIT License</strong>. See <a href="LICENSE">LICENSE</a> for details.</p>
