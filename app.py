import os
import uuid
import base64
import streamlit as st
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
st.title("DocPilot: The MultiModal RAG Assistant")
st.write("Upload a PDF to ask questions about its content.")

# Helper Functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_documents(texts, image_base64s, image_summaries, tables, table_summaries):
    documents = []
    #Add text
    for content in texts:
        doc_id = str(uuid.uuid4())
        doc = Document(
            page_content=content,
            metadata={
                "id": doc_id,
                "type": "text",
                "original_content": content
            }
        )
        documents.append(doc)
        
    # Add table summary
    for content, summary in zip(tables, table_summaries):
        doc_id = str(uuid.uuid4())
        doc = Document(
            page_content=summary,
            metadata={
                "id": doc_id,
                "type": "text",
                "original_content": content
            }
        )
        documents.append(doc)
        
    # Add image summaries
    for b64, summary in zip(image_base64s, image_summaries):
        doc_id = str(uuid.uuid4())
        doc = Document(
            page_content=summary,
            metadata={
                "id": doc_id,
                "type": "image",
                "original_content": b64
            }
        )
        documents.append(doc)
    return documents


def process_uploaded_file(uploaded_file, llm, vision_llm):
    temp_dir = "./uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Partition the PDF
    image_output_dir = "./raw_elements"
    os.makedirs(image_output_dir, exist_ok=True)
    
    with st.spinner("Step 1/4: Extracting elements from the PDF..."):
        raw_elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_output_dir=image_output_dir
        )

        texts, tables = [], []
        for element in raw_elements:
            if "unstructured.documents.elements.Text" in str(type(element)):
                texts.append(str(element))
            elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
                texts.append(str(element))
            elif "unstructured.documents.elements.ListItem" in str(type(element)):
                texts.append(str(element))
            elif "unstructured.documents.elements.FigureCaption" in str(type(element)):
                texts.append(str(element))
            elif "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))

    # Summarize tables
    prompt_text = "You are an assistant tasked with summarizing text for retrieval. Give a concise summary of the following content that is well optimized for retrieval:\n---\n{element}"
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = prompt | llm | StrOutputParser()
    
    with st.spinner("Step 2/4: Summarizing text and tables..."):
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    # Summarize Images
    with st.spinner("Step 3/4: Summarizing images..."):
        image_base64s = []
        image_summaries = []
        image_paths = sorted([os.path.join(image_output_dir, f) for f in os.listdir(image_output_dir) if f.endswith(".jpg")])
        
        for img_path in image_paths:
            base64_image = encode_image(img_path)
            image_base64s.append(base64_image)
            
            image_message = HumanMessage(
                content=[
                    {"type": "text", "text": "Summarize this image for retrieval. Describe its key elements and purpose concisely."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            summary = vision_llm.invoke([image_message]).content
            image_summaries.append(summary)

    # Create documents and build the vector store
    with st.spinner("Step 4/4: Creating vector store..."):
        documents = create_documents(texts, image_base64s, image_summaries, tables, table_summaries)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    
    st.session_state.db = vectorstore
    st.success("File processed successfully! You can now ask questions.")


# --- Reranking with Cross-Encoder ---
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2') 
cross_encoder = load_cross_encoder()

def rerank_documents(query, retrieved_docs, top_n=5):
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = list(zip(scores, retrieved_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    return [doc for score, doc in scored_docs[:top_n]]

# File upload
with st.sidebar:
 
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("Process Document"):
            llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
            vision_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
            process_uploaded_file(uploaded_file, llm, vision_llm)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message and message["images"]:
            for img_b64 in message["images"]:
                st.image(base64.b64decode(img_b64), width=300)

prompt = st.chat_input("Ask a question about the document...")

if prompt and prompt.strip() != "":
    if "db" not in st.session_state or st.session_state.db is None:
        st.warning("Please upload and process a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                db = st.session_state.db
                
                # Retrieval
                retrieved_docs = db.similarity_search(prompt, k=10)
                
                # Reranking
                reranked_docs = rerank_documents(prompt, retrieved_docs)
                
                # Build context
                context = ""
                relevant_images = []
                for doc in reranked_docs:
                    if doc.metadata["type"] == "text":
                        context += doc.metadata["original_content"] + "\n---\n"
                    elif doc.metadata["type"] == "table":
                        context += doc.page_content + "\n---\n"    
                    elif doc.metadata["type"] == "image":
                        context += doc.page_content + "\n---\n"
                        relevant_images.append(doc.metadata["original_content"])
                
                # Generate Answer
                rag_prompt_template = """
                You are an expert AI assistant. Answer the question based ONLY on the following context, which can include text, tables, and image summaries:
                CONTEXT:
                {context}
                QUESTION: {question}
                If the context does not contain the answer, say "Sorry, I don't have enough information to answer that."
                Answer:
                """
                rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
                llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
                rag_chain = rag_prompt | llm | StrOutputParser()
                
                response = rag_chain.invoke({"context": context, "question": prompt})
                
                st.markdown(response)
                if relevant_images:
                    st.write("Relevant images from the document:")
                    for img_b64 in relevant_images:
                        st.image(base64.b64decode(img_b64), width=300)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "images": relevant_images
                })