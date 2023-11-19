import streamlit as st
from PyPDF2 import PdfReader

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Fix bug "no attribute 'verbose'" https://github.com/hwchase17/langchain/issues/4164
langchain.verbose = False


def main():
    # Callback to stream output to stdout, can be removed
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Download a LlamaCpp model manually from https://huggingface.co and place it under same folder as this
    # using example model "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q6_K.gguf"
    llm = LlamaCpp(
        model_path="./tinyllama-1.1b-chat-v0.3.Q6_0.gguf",
        temperature=0.6,
        max_tokens=2000,
        stop=["<|im_start|>user"], # This is part of the prompt format of the model
        callback_manager=callback_manager,
        verbose=True,
        top_p=1,
        n_ctx=4096,
        n_batch=512,
    )

    # Load question answering chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Patching qa_chain prompt template to better suit prompt format from chosen model
    if "Helpful Answer:" in chain.llm_chain.prompt.template:
        chain.llm_chain.prompt.template = (
            f"<|im_start|>user{chain.llm_chain.prompt.template}".replace(
                "Helpful Answer:", "<|im_end|>\n<|im_start|>assistant"
            )
        )

    # Page titles
    st.set_page_config(page_title="PDF Consulting AI", page_icon=":books:")
    st.header(":books: PDF Consulting AI :books:")
    # Ingreso de PDFs
    pdf_docs = st.file_uploader("Upload your PDFs here:", type=["pdf"], accept_multiple_files=True)

    if pdf_docs:
        text = ""
        # loop through PDF
        for pdf in pdf_docs:
            # initialize PdfReader
            pdf_reader = PdfReader(pdf)
            # loop through pages and concatenate to text
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Split text in chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Automatically download embedding model from https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

        # Create in-memory Qdrant instance
        knowledge_base = Qdrant.from_texts(
            chunks,
            embeddings,
            location=":memory:",
            collection_name="doc_chunks",
        )

        user_question = st.text_input("Ask your question next:")

        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=4)

            # Grab and print response
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)


if __name__ == "__main__":
    main()
