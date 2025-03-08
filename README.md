# langchain-rag-pdf-url-chatbot

This repository contains a **Retrieval-Augmented Generation (RAG) Chatbot** built using **LangChain, OpenAI, and Streamlit**. The chatbot allows users to **upload a PDF via a URL**, process it into vector embeddings using **ChromaDB**, and interact with the document using a **chat interface**.

## 🚀 Features

-   **Load PDFs from URLs** and extract their content.
-   **Split documents into chunks** for efficient retrieval.
-   **Store embeddings in ChromaDB** for fast similarity search.
-   **Use OpenAI's GPT-4 Turbo** to generate answers from retrieved document context.
-   **Streamlit-based UI** for an interactive Q&A experience.

## 📌 How It Works

1. **User enters a PDF URL** → The document is fetched and split into chunks.
2. **Embeddings are generated** using OpenAI's embedding model and stored in ChromaDB.
3. **User asks a question** → The chatbot retrieves relevant chunks and generates a response.
4. **A chat history is maintained** for context-aware responses.

## 🛠️ Technologies Used

-   **Python**
-   **Streamlit** (for UI)
-   **LangChain** (for document processing & RAG pipeline)
-   **OpenAI API** (for embeddings & chat completions)
-   **ChromaDB** (for vector storage)

## 🔧 Setup

1. Clone the repo:
    ```bash
    git clone git@github.com:AlgoMart/langchain-rag-pdf-url-chatbot.git
    cd langchain-rag-pdf-url-chatbot
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up OpenAI API key in `.env`:
    ```
    OPENAI_API_KEY=your_api_key
    ```
4. Run the chatbot:
    ```bash
    streamlit run chatbot.py
    ```

## 📚 Use Cases

-   Academic research assistance
-   Legal document Q&A
-   Interactive book summaries
-   Enterprise knowledge retrieval

### ⭐ Star the repo if you find it useful! 🚀

## Chatbot Images

#### Screen as soon as application starts

![Screen as soon as application starts](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*RGuirLr1K6j-ABXzvwiNEw.png)

#### Screen when you start loading document

![Screen when you start loading document](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*sPyvZ-JDJuor1K1khBIqeA.png)

#### Screen where you can chat

![Screen where you can chat](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vIQ_2GxIFWSdlnAApIjQ2g.png)

#### Screen where LLM is thinking

![Screen where LLM is thinking](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3v7vRNAx21AJAztaHVtWgg.png)

#### Screen with LLM response

![Screen with LLM response](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ZAjy38Bu5fldsrmie0zqmg.png)
