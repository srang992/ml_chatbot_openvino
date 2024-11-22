import streamlit as st
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from message import Message
from prompt import prompt_qwen
from utils import *


response = ""

st.set_page_config(page_title="Machine Learning Chatbot", page_icon="🤖")
st.title("🤖 Machine Learning Chatbot")
st.write(":grey[This chatbot is created for solving machine learning queries and the model is optimized using \
    **OpenVINO Toolkit**.]")


@st.cache_resource(ttl="1d")
def load_models():
    embed_model = OpenVINOEmbeddings(
    model_name_or_path=embed_model_dir,
    model_kwargs=model_kwargs,
    encode_kwargs = encode_kwargs
    )

    reranker = OpenVINOReranker(
        model_name_or_path=reranker_model_dir,
        model_kwargs=model_kwargs,
        top_n=2,
    )
    
    ov_llm = HuggingFacePipeline.from_model_id(
    model_id=chat_model_dir,
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": "CPU", 
                  "ov_config": ov_config, 
                  "temperature": 0.0, 
                  "max_length": 512,
                  "repetition_penalty": 1.1},
    )

    return embed_model, ov_llm, reranker


if MESSAGES not in st.session_state:
    text = "Hello! How can I assist you today?"
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload=text)]
        
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

embed_model, ov_llm, reranker = load_models()
vectorstore = FAISS.load_local("vectorstore", embed_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever_with_rerank = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
llm = ov_llm.bind(**gen_config)
qna_chain = create_stuff_documents_chain(llm, prompt_qwen)
rag_chain = create_retrieval_chain(retriever=retriever_with_rerank, combine_docs_chain=qna_chain)
chain = rag_chain.pick("answer")

if user_msg:=st.chat_input("What is Machine Learning?"):
    st.session_state[MESSAGES].append(Message(actor=USER, payload=user_msg))
    st.chat_message(USER).write(user_msg)

    def gen_answer():
        global response
        for chunk in chain.stream({"input": user_msg}):
            yield chunk
            response += chunk

    with st.chat_message(ASSISTANT):
        st.write_stream(gen_answer)

    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    response = ""