import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from chatbot import final_chain 

st.set_page_config(page_title="Simple Graph Chatbot")
st.title("RAG Chatbot for Build Fast with AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)

if user_query := st.chat_input("Ask me anything about Build Fast with AI..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    try:
        response = final_chain.invoke({
            "question": user_query,
            "chat_history": [(m.content, n.content) for m, n in zip(
                st.session_state.messages[::2], st.session_state.messages[1::2]
            )] if len(st.session_state.messages) >= 2 else []
        })

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append(AIMessage(content=response))
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
