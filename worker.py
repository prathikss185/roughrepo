import os
import torch
import traceback
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load HF token from external file and set env variable
with open("hf_token.txt", "r") as f:
    hf_token = f.read().strip()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

def init_llm():
    global llm_hub, embeddings

    model_id = "meta-llama/Llama-2-7b-chat-hf"

    llm_hub = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=512,
        max_length=1024
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

def process_document(document_path):
    global conversation_retrieval_chain

    loader = PyPDFLoader(document_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(texts, embedding=embeddings)

    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )
    print("Document processed and retrieval chain initialized.")

def process_prompt(prompt):
    global chat_history

    if not conversation_retrieval_chain:
        return "Please upload a document first."

    try:
        # Use invoke() as per deprecation warning
        output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
        answer = output["result"]
        chat_history.append((prompt, answer))
        return answer
    except Exception as e:
        print("Error in process_prompt:", e)
        traceback.print_exc()
        return "Sorry, there was an error processing your question."

def test_chain():
    if not conversation_retrieval_chain:
        print("Chain is not initialized! Please upload and process a document first.")
        return

    test_question = "What is this document about?"
    try:
        output = conversation_retrieval_chain.invoke({"question": test_question, "chat_history": []})
        print("Test output:", output["result"])
    except Exception as e:
        print("Error testing chain:", e)
        traceback.print_exc()

# Initialize LLM on module import
init_llm()

