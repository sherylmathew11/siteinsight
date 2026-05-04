# 💡 SiteInsight

**SiteInsight** is a real-time AI tool that transforms any webpage into a conversational knowledge base. By simply pasting a URL, the application extracts and cleans the text content, embeds it into a semantic search index, and allows you to chat directly with the webpage using advanced Retrieval-Augmented Generation (RAG).

---

## 🚀 Core Capabilities

* **Instant Link Extraction & Cleaning:** Automatically scrapes webpage text using `BeautifulSoup`, cleanly filtering out non-content elements such as ads, headers, footers, scripts, and sidebars.
* **Smart Contextual Chunking:** Splits heavy articles or long-form documentation into highly manageable, semantically sound text chunks with overlap.
* **Semantic Search & Vector Memory:** Generates accurate context embeddings via `OpenAIEmbeddings` and builds a fast, high-performance in-memory local vector database using `FAISS`.
* **Advanced Reranking Pipeline:** Retrieves relevant content using Maximal Marginal Relevance (MMR) and then applies a secondary reranking layer using `OpenAI GPT-4o` to surface the most contextually relevant information.
* **Accurate Contextual QA:** Formulates professional responses exclusively using extracted source text via `OpenAI GPT-4o`, minimizing hallucinations by explicitly indicating if the answer cannot be found in the context.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **AI & Orchestration:** LangChain (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`, `langchain-classic`)
* **LLM & Embeddings:** OpenAI (GPT-4o & Text Embeddings)
* **Vector Store:** FAISS (CPU)
* **Web Scraping:** Requests & BeautifulSoup4

---

## 💻 Local Setup & Installation

Follow these instructions to configure the environment and run SiteInsight on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/siteinsight.git](https://github.com/your-username/siteinsight.git)
cd siteinsight
```

### 2. Get your OpenAPI API Key

You need an **OpenAI API key** to generate embeddings and power responses. To get your OpenAI API Key, go to OpenAI Platform: https://platform.openai.com/  

### 3. Installation

It is recommended to use a **Python virtual environment** to avoid package conflicts.

#### Create a virtual environment

```bash
python -m venv venv
```

#### Activate the virtual environment:

- **Windows**
```bash
venv\Scripts\activate
```

- **Mac / Linux**
```bash
source venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Streamlit app locally:

```bash
streamlit run app.py
```

---

## 📖 Step-by-Step Usage Guide

Once the Streamlit application is running in your browser:

1. **Add API Key:** Enter your **OpenAI API Key** in the sidebar.
2. **Enter a URL:** Paste the specific webpage URL you wish to analyze in the input field.
3. **Index Content:** Click **"Extract and Index Data"** to scrape, chunk, and build the local index using OpenAI embeddings.
4. **Ask Questions:** Use the main chat box to ask questions directly. The AI will strictly use the webpage context to respond.
5. **Review Sources:** Expand the **"Show Retrieved Source Chunks"** dropdown to inspect the text used for generating the answer.
