"""
STEP 6 - CHAT UI (Streamlit)
----------------------------
Run:   streamlit run rag/app.py
Then open the link it prints (usually http://localhost:8501).

Make sure you ran step 3 first so chroma_db/ exists.
"""
import sys
from pathlib import Path

# Lets "streamlit run rag/app.py" find the rag package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from rag.step5_generate import answer

st.set_page_config(page_title="Marwadi University Assistant", page_icon="🎓")
st.title("🎓 Marwadi University Assistant")
st.caption("Ask about fees, timings, HODs, hostel, exams, admissions...")

# st.session_state keeps data between reruns (Streamlit reruns the script on every message)
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role", "content", "sources"}]

# Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

question = st.chat_input("Type your question...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Only send plain question/answer text as history (last 6 messages is enough)
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[-6:]]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, hits = answer(question, history)
        sources = sorted({h["source"] for h in hits})
        st.markdown(reply)
        st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
