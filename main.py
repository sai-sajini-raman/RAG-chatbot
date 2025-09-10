"""Streamlit app:
- Chat UI
- Conversation memory (per session)
- New Chat button
- Simple monitoring dashboard (counts, avg latency)
"""
import streamlit as st
from rag_pipeline import answer_query
from monitoring import log
from config import CSV_LOG_PATH
import time
import pandas as pd

st.set_page_config(page_title="RAG Chatbot (Excel)", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, text, sources}

st.title("📚 RAG Chatbot — MAL-Food-SC Knowledge Base")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("🔄 New Chat"):
        st.session_state.messages = []
    user_query = st.text_input("Ask a question (about the uploaded Excel knowledge):", key="query_input")
    if st.button("Send") or (user_query and st.session_state.get('last_sent') != user_query):
        if user_query:
            st.session_state.last_sent = user_query
            st.session_state.messages.append({"role":"user","text":user_query})
            with st.spinner("Retrieving and generating answer..."):
                start = time.time()
                res = answer_query(user_query)
                latency = res.get("latency")
                answer = res.get("answer")
                sources = res.get("sources", [])
                st.session_state.messages.append({"role":"assistant","text":answer, "sources": sources})
                # log
                log(user_query, answer, sources, latency)
with col2:
    st.subheader("Monitoring")
    try:
        df = pd.read_csv(CSV_LOG_PATH)
        total = len(df)
        avg_latency = df['latency'].mean()
        st.metric("Total queries logged", total)
        st.metric("Average latency (s)", f"{avg_latency:.2f}")
        st.dataframe(df.tail(20))
    except Exception:
        st.info("No logs yet. Run queries to build metrics.")

st.markdown("---")
# Render chat
for i, m in enumerate(st.session_state.messages):
    if m['role'] == 'user':
        st.markdown(f"**You:** {m['text']}")
    else:
        st.markdown(f"**Assistant:** {m['text']}")
        if m.get('sources'):
            st.markdown("**Sources:**")
            for s in m['sources']:
                st.markdown(f"- sheet: {s.get('sheet')}, row: {s.get('row_index')}, chunk: {s.get('chunk_id')}")

# allow rating latest assistant response
if st.session_state.messages:
    last = st.session_state.messages[-1]
    if last['role'] == 'assistant':
        st.markdown("### Rate the last answer")
        col_a, col_b = st.columns(2)
        if col_a.button("👍 Good", key="good_btn"):
            # update logs with quality 'good' (simple approach: append a new log entry)
            user_q = st.session_state.messages[-2]['text'] if len(st.session_state.messages)>=2 else ""
            log(user_q, last['text'], last.get('sources', []), 0.0, quality='good')
            st.success("Thanks for the feedback!")
        if col_b.button("👎 Bad", key="bad_btn"):
            user_q = st.session_state.messages[-2]['text'] if len(st.session_state.messages)>=2 else ""
            log(user_q, last['text'], last.get('sources', []), 0.0, quality='bad')
            st.success("Thanks — feedback noted.")
