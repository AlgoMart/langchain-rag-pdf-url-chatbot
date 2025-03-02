from typing import Callable, List

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGChatbot:
    def __init__(self):
        load_dotenv()

    def load_documents_from_url(self, url: str) -> List[Document]:
        """
        Loads PDF documents from the given URL.
        """
        documents = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            loader = PyPDFLoader(url, headers=headers)
            documents.extend(loader.load())
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching the PDF: {e}")
        return documents

    def split_documents_using_text_splitter(self, documents: List[Document]) -> List[Document]:
        """
        Splits the loaded documents into chunks based on the splitter configuration.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document], collection_name: str = "my_collection") -> Chroma:
        """
        Creates and persists a vector store based on the document chunks.
        """
        # document_embeddings = self.embeddings.embed_documents([split.page_content for split in documents])
        embeddings = OpenAIEmbeddings()

        # Initialize ChromaDB
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )

        # Add to vector store in a batch
        batch_size = 50
        batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

        progress_bar = st.progress(0)
        status_text = st.empty()
        for index, batch in enumerate(batches):
            vector_store.add_documents(batch)
            progress = (index + 1) / len(batches)
            print(f"Percentage done: {progress}")

            # Update progress bar and text
            progress_bar.progress(progress)
            status_text.text(f"Processing: {int(progress * 100)}% done")

        return vector_store

    def get_retriever(self, vector_store: Chroma, k: int = 10):
        """
        Retrieves the top k results from the vector store.
        """
        return vector_store.as_retriever(search_kwargs={"k": k})

    def docs_to_string(self, docs: List[Document]) -> str:
        """
        Converts a list of documents into a single concatenated string.
        """
        return "\n\n".join([doc.page_content for doc in docs])

    def create_rag_chain(
        self, retriever: VectorStoreRetriever, docs_to_string_func: Callable[[List[Document]], str]
    ) -> Runnable:
        """
        Creates the RAG chain for answering the questions.
        """
        # Create the Prompt Template
        template = """
            Answer the question based only on the following context:
            {context}
            Question: {question}
            Answer:
        """
        prompt = PromptTemplate.from_template(template)

        # Initalize the LLM
        llm = ChatOpenAI(model="gpt-4-turbo")

        # Chain
        return (
            {"context": retriever | docs_to_string_func, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask_question(self, question: str, rag_chain: Runnable) -> str:
        """
        Invokes the RAG chain with the provided question.
        """
        return rag_chain.invoke(question)


def initialize_session() -> None:
    """Initialize session state variables."""
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_document(chatbot: RAGChatbot, pdf_url: str) -> None:
    """Load and process the document from the given URL."""
    if not pdf_url:
        st.warning("Please enter a valid PDF URL.")
        return

    with st.spinner("Loading document..."):
        documents = chatbot.load_documents_from_url(pdf_url)
        if not documents:
            st.error("Failed to load the document. Please check the URL and try again.")
            return

        splits = chatbot.split_documents_using_text_splitter(documents)
        vector_store = chatbot.create_vector_store(splits)
        retriever = chatbot.get_retriever(vector_store)

        st.session_state.rag_chain = chatbot.create_rag_chain(retriever, chatbot.docs_to_string)
        st.session_state.document_loaded = True
        st.success(f"Loaded and processed {len(splits)} document chunks.")
        st.rerun()


def chat_interface(chatbot: RAGChatbot) -> None:
    """Render the chat interface for user interaction."""
    st.subheader("💬 Chat with the Document")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Thinking..."):
                chat_history = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
                )
                full_prompt = f"Chat History:\n{chat_history}\n\nNew Question: {prompt}"

                # **Ask the chatbot with chat history**
                response = (
                    chatbot.ask_question(full_prompt, st.session_state.rag_chain)
                    if st.session_state.rag_chain
                    else "Please load a document first."
                )
            response_container.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    st.title("📄 RAG Chatbot with Streamlit")
    chatbot = RAGChatbot()
    initialize_session()

    pdf_url = st.text_input("Enter PDF URL:", disabled=st.session_state.document_loaded)
    if st.button("Load Document", disabled=st.session_state.document_loaded):
        load_document(chatbot, pdf_url)

    if st.session_state.rag_chain:
        chat_interface(chatbot)
