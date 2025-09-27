# DocPilot: The MultiModal RAG Assistant 📚
<img width="1910" height="928" alt="Screenshot 2025-09-27 143825" src="https://github.com/user-attachments/assets/4ccf9d54-628b-408b-a1ce-03879cc47d1e" />

DocPilot is an interactive web application built with Streamlit that allows you to "chat" with your PDF documents. It leverages a powerful multimodal Retrieval-Augmented Generation (RAG) pipeline to understand and answer questions about the text, tables, and images within your files.

## Features

- **Multimodal Understanding**: Extracts and processes not just text, but also tables and images from your PDFs.  
- **Advanced RAG Pipeline**: Uses a sophisticated two-stage process for retrieving information:  
  1. **Efficient Vector Search**: Quickly finds potentially relevant document chunks using FAISS.  
  2. **Cross-Encoder Reranking**: Intelligently reranks the initial results to find the most accurate context for your question.  
- **AI-Powered Summarization**: Leverages OpenAI's GPT-4o to create concise, retrieval-optimized summaries of images and tables.  
- **Interactive Chat Interface**: A user-friendly, chat-based interface to ask questions and receive answers, complete with relevant images from the document.  
- **Built with State-of-the-Art Tools**: Powered by LangChain, Streamlit, OpenAI, and Sentence Transformers.  

---

## How It Works

DocPilot employs a multi-step process to provide accurate answers from your documents:

1. **PDF Parsing**: When you upload a PDF, the application uses the `unstructured` library to partition it into distinct elements: text, tables, and images.  
2. **Content Summarization**:  
   - **Text** is stored directly.  
   - **Tables** and **images** are summarized by the powerful GPT-4o model to create text descriptions that are optimized for semantic search.  
3. **Vector Store Creation**: The text and the summaries are embedded into a vector space and stored in a FAISS vector store. This allows for rapid similarity searches.  
4. **Retrieval and Reranking**: When you ask a question:  
   - The app first performs a quick search on the vector store to retrieve the top 10 most relevant chunks.  
   - These 10 chunks are then passed to a Cross-Encoder model, which performs a more computationally intensive but highly accurate reranking to select the best possible context.  
5. **Answer Generation**: The final, reranked context is combined with your original question in a prompt that is sent to GPT-4o, which generates a comprehensive answer based only on the provided information.  

---

## Installation and Setup

Follow these steps to get DocPilot running on your local machine.

### Prerequisites

- Python 3.8 or higher  
- An OpenAI API Key  

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/docpilot.git
cd docpilot
````

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

Create a `.env` file in the root of the project directory and add your OpenAI API key:

```
OPENAI_API_KEY="your-api-key-here"
```

---

## How to Run the Application

With the setup complete, you can now run the Streamlit application.

1. Open your terminal in the project directory.

2. Run the following command:

   ```bash
   streamlit run app.py
   ```

3. The application will open in a new tab in your web browser. Now you can upload a PDF, click "Process Document," and start asking questions!
