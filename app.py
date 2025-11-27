import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from utils.functions import create_document
from prompts.system_prompts import *
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_PROJECT']='PDF Q&A Chatbot memory'
# os.environ['LANGCHAIN_TRACING_V2']='true'

llm=ChatGroq(model='llama-3.1-8b-instant')
embeddings=HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')


if "messages" not in st.session_state:
    st.session_state.messages = []

if "clarification_count" not in st.session_state:
    st.session_state.clarification_count = 0


st.sidebar.title('Give your Information')
full_name=st.sidebar.text_input('name')
Email_Address=st.sidebar.text_input('email address')
phone_no=st.sidebar.text_input('phone no')
years_of_exp=st.sidebar.text_input('experience')
desired_positions=st.sidebar.text_input('Desired position(s)')
current_loc=st.sidebar.text_input('current location')
tech_stack=st.sidebar.text_input('specify your tech stack, including programming languages, frameworks, databases, and tools you are proficient in.')
st.set_page_config(page_title="TalentScout Hiring Assistant")



if st.sidebar.button("Submit"):
    docs = create_document(full_name, years_of_exp, desired_positions, tech_stack)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    st.session_state["retriever"] = retriever
    st.session_state["candidate_ready"] = True
    history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=history_aware_retriever_prompt,
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=rag_answer_prompt,
    )

    base_rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    def get_session_history(session_id: str):
        if "lc_histories" not in st.session_state:
            st.session_state["lc_histories"] = {}
        if session_id not in st.session_state["lc_histories"]:
            st.session_state["lc_histories"][session_id] = ChatMessageHistory()
        return st.session_state["lc_histories"][session_id]

    rag_chain_with_history = RunnableWithMessageHistory(
        base_rag_chain,          # result of create_retrieval_chain
        get_session_history,     # the function above
        input_messages_key="input",       # key in your invoke() dict
        history_messages_key="chat_history",  # matches your prompts' MessagesPlaceholder
    )

    st.session_state["rag_chain"] = rag_chain_with_history

    if not st.session_state.messages:
        greeting = (
            f"Hello {full_name or 'there'}, I got your application.\n\n"
            "I'll ask a few follow‑up questions and then some technical questions "
            "based on your tech stack. Please Acknowledge by saying 'Okay'."
        )
        st.session_state.messages.append({"role": "assistant", "content": greeting})

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask something or continue the interview...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    if not st.session_state.get("candidate_ready"):
        with st.chat_message("assistant"):
            msg = "Please submit your basic information in the sidebar first."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.write(msg)
    else:
        rag_chain = st.session_state["rag_chain"]
        with st.chat_message("assistant"):
            result = rag_chain.invoke(
                {
                    "input": user_input,
                    "clarification_count": st.session_state.clarification_count,
                },
                config={"configurable": {"session_id": "default"}},
            )
            answer = result["answer"] if isinstance(result, dict) else result
            is_clarifying = "[CLARIFY]" in answer

            if is_clarifying:
                st.session_state.clarification_count += 1

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)

    
    
    
