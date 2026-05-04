import os
import requests
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS


# ----------------------------
# Config
# ----------------------------
load_dotenv()

st.set_page_config(page_title="SiteInsight", page_icon="💡", layout="wide")
st.title("💡 SiteInsight")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")

    openai_api_key = st.text_input(
        "Enter OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )

    st.markdown("---")
    st.info(
        "Paste a link to instantly build a knowledge base and chat with the webpage's content." \
        "\n" \
        "1. Enter an OpenAI API Key.\n" \
        "2. Input a webpage URL.\n" \
        "3. Click 'Extract and Index Data' to build the knowledge base.\n" \
        "4. Ask questions about the content of the webpage!"
    )

# ----------------------------
#  Scraping / ingestion
# ----------------------------
def clean_webpage(url):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=20
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "svg"
    ]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("body")
    )

    title = soup.title.string.strip() if soup.title else "Untitled"

    text = main.get_text(separator="\n", strip=True)
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    return title, text


# ----------------------------
# Build vector DB
# ----------------------------
@st.cache_resource(show_spinner=False)
def setup_vector_db(url, api_key):
    if not api_key:
        return None

    title, cleaned_text = clean_webpage(url)

    base_doc = Document(
        page_content=cleaned_text,
        metadata={
            "source": url,
            "title": title
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents([base_doc])

    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        enriched_chunks.append(chunk)

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key
    )

    vector_db = FAISS.from_documents(
        enriched_chunks,
        embeddings
    )

    return vector_db


# ----------------------------
# Reranker
# ----------------------------
def rerank_docs(question, docs, api_key):
    if not docs:
        return docs

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key
    )

    scored = []

    for doc in docs:
        prompt = f"""
Rate relevance from 1-10.

Question:
{question}

Document:
{doc.page_content[:2500]}

Only return a number.
"""

        try:
            score = llm.invoke(prompt).content.strip()
            score = float(score)
        except Exception:
            score = 0

        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for score, doc in scored[:5]]


# ----------------------------
# URL input
# ----------------------------
source_url = st.text_input(
    "Enter a Website URL:",
    value="https://en.wikipedia.org/wiki/Machine_learning"
)


# ---------------------------- 
# Build Knowledge Base
# ----------------------------
if st.button("Extract and Index Data"):
    if not openai_api_key:
        st.warning("Please enter OpenAI API key.")
    elif not source_url:
        st.warning("Please enter valid URL.")
    else:
        with st.spinner("Building knowledge base..."):
            vectorstore = setup_vector_db(
                source_url,
                openai_api_key
            )

            if vectorstore:
                st.session_state["vector_db"] = vectorstore
                st.session_state["chat_history"] = []
                st.success("Knowledge base indexed successfully.")


# ----------------------------
# Chat
# ----------------------------
st.markdown("---")
st.header("Ask your Knowledge Base")

user_question = st.text_input(
    "What would you like to know?"
)


# ----------------------------
# Query flow
# ----------------------------
if user_question:
    if "vector_db" not in st.session_state:
        st.error("Please extract and index data first.")

    elif not openai_api_key:
        st.error("Please enter OpenAI API key.")

    else:
        with st.spinner("Retrieving context and generating answer..."):

            try:
                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []

                chat_history_text = "\n".join(
                    [
                        f"User: {x['question']}\nAssistant: {x['answer']}"
                        for x in st.session_state["chat_history"]
                    ]
                )

                # OpenAI LLM
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                    openai_api_key=openai_api_key
                )

                # MMR retrieval
                retriever = st.session_state["vector_db"].as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 10,
                        "fetch_k": 30,
                        "lambda_mult": 0.5
                    }
                )

                retrieved_docs = retriever.invoke(user_question)

                # Rerank
                final_docs = rerank_docs(
                    user_question,
                    retrieved_docs,
                    openai_api_key
                )

                context = "\n\n".join(
                    [
                        f"[Source: {doc.metadata.get('title')} | Chunk: {doc.metadata.get('chunk_id')}]\n{doc.page_content}"
                        for doc in final_docs
                    ]
                )

                prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the provided context.

If context is insufficient, say you do not know.

Conversation history:
{history}

Context:
{context}

Question:
{input}

Answer:
""")

                chain = create_stuff_documents_chain(
                    llm,
                    prompt
                )

                response = chain.invoke({
                    "history": chat_history_text,
                    "context": final_docs,
                    "input": user_question
                })

                st.subheader("Answer:")
                st.write(response)

                st.session_state["chat_history"].append({
                    "question": user_question,
                    "answer": response
                })

                with st.expander("Show Retrieved Source Chunks"):
                    for i, doc in enumerate(final_docs, 1):
                        st.markdown(
                            f"### Chunk {i} "
                            f"(Source: {doc.metadata.get('title')}, "
                            f"Chunk ID: {doc.metadata.get('chunk_id')})"
                        )
                        st.write(doc.page_content)
                        st.markdown("---")

            except Exception as e:
                st.error(f"Error: {e}")