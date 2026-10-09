"""Step 3: simple chat interface. Run with:  streamlit run app.py"""
import streamlit as st

from rag import answer

st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("📄 Chat with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show earlier messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                reply, hits = answer(question, st.session_state.messages)
            st.markdown(reply)
            if hits:
                with st.expander("Sources used"):
                    for i, h in enumerate(hits, 1):
                        st.markdown(f"**[{i}] {h['source']} (page {h['page']})**")
                        st.caption(h["text"])
        except Exception as e:
            reply = f"Error: {e}"
            st.error(reply)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": reply})
