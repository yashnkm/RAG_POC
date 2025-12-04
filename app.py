"""
RAG Demo - Streamlit UI
Step-by-step visualization of how RAG works
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

# ----- Page Config -----
st.set_page_config(page_title="RAG Demo", page_icon="🔍", layout="wide")

# ----- Styles -----
st.markdown("""
<style>
    .step-box { background-color: #ffffff; color: #000000; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 5px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .step-box p, .step-box strong { color: #000000; }
    .step-header { color: #1f77b4; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
    .chunk-box { background-color: #ffffff; color: #000000; border-radius: 5px; padding: 10px; margin: 5px 0; font-size: 0.9em; border: 1px solid #28a745; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .chunk-box strong, .chunk-box small { color: #000000; }
    .embedding-box { background-color: #ffffff; color: #000000; border-radius: 5px; padding: 10px; margin: 5px 0; font-family: monospace; font-size: 0.8em; border: 1px solid #ffc107; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .embedding-box strong, .embedding-box small { color: #000000; }
    .retrieval-box { background-color: #ffffff; color: #000000; border-radius: 5px; padding: 10px; margin: 5px 0; border: 1px solid #28a745; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .retrieval-box strong { color: #000000; }
    .score-badge { background-color: #1f77b4; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; }
    .answer-box { background-color: #ffffff; color: #000000; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 5px solid #004085; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .answer-box strong { color: #000000; }
    .agent-box { background-color: #ffffff; color: #000000; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 5px solid #6f42c1; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .agent-box strong, .agent-box p { color: #000000; }
    .tool-call-box { background-color: #ffffff; color: #000000; border-radius: 5px; padding: 10px; margin: 5px 0; border: 2px dashed #6f42c1; }
    .tool-call-box strong, .tool-call-box small { color: #000000; }
</style>
""", unsafe_allow_html=True)

# ----- Title -----
st.title("🔍 RAG Demo: Step-by-Step")
st.markdown("*Watch how RAG processes documents and answers questions*")

# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key set!")

    st.markdown("---")
    st.markdown("### 🎯 Try These Questions")
    st.markdown("- How many days of annual leave?")
    st.markdown("- What is the password policy?")
    st.markdown("- What is the daily meal allowance?")
    st.markdown("- What is 2+2? *(no RAG needed)*")

# ----- Session State -----
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
if "ready" not in st.session_state:
    st.session_state.ready = False

rag = st.session_state.rag

# ----- Tabs -----
tab1, tab2, tab3 = st.tabs(["📥 Document Processing", "🔍 Basic RAG", "🤖 Agentic RAG"])

# ===== TAB 1: Document Processing =====
with tab1:
    if not api_key:
        st.warning("⚠️ Enter your Google API Key in the sidebar")
    else:
        # STEP 1: Load
        st.markdown("### 📄 STEP 1: Load Documents")
        if st.button("Load Documents", type="primary"):
            with st.spinner("Loading..."):
                docs = rag.load_documents()
                for doc in docs:
                    time.sleep(0.3)
                    st.markdown(f'''
                    <div class="step-box">
                        <div class="step-header">✅ {doc.metadata["source"]}</div>
                        <p><strong>Size:</strong> {len(doc.page_content):,} characters</p>
                        <p>{doc.page_content[:150]}...</p>
                    </div>
                    ''', unsafe_allow_html=True)
                st.success(f"Loaded {len(docs)} documents")

        # STEP 2: Chunk
        st.markdown("### ✂️ STEP 2: Chunk Documents")
        col1, col2 = st.columns(2)
        chunk_size = col1.slider("Chunk Size", 200, 1000, 500)
        chunk_overlap = col2.slider("Overlap", 0, 200, 50)

        if st.button("Create Chunks", type="primary"):
            if not rag.documents:
                st.error("Load documents first!")
            else:
                with st.spinner("Chunking..."):
                    chunks = rag.chunk_documents(chunk_size, chunk_overlap)
                    st.markdown(f"**Created {len(chunks)} chunks**")
                    for i, chunk in enumerate(chunks):
                        st.markdown(f'''
                        <div class="chunk-box">
                            <strong>Chunk {i+1}</strong> | {chunk.metadata["source"]} | {len(chunk.page_content)} chars
                            <hr><small>{chunk.page_content[:200]}...</small>
                        </div>
                        ''', unsafe_allow_html=True)
                    st.success(f"Created {len(chunks)} chunks")

        # STEP 3: Embed
        st.markdown("### 🧮 STEP 3: Create Embeddings")
        if st.button("Create Embeddings & Vector Store", type="primary"):
            if not rag.chunks:
                st.error("Create chunks first!")
            else:
                with st.spinner("Creating embeddings..."):
                    st.markdown("**Sample Embeddings:**")
                    for i, chunk in enumerate(rag.chunks[:3]):
                        vec = rag.get_embedding(chunk.page_content[:100])
                        vec_preview = ", ".join([f"{v:.3f}" for v in vec[:8]])
                        st.markdown(f'''
                        <div class="embedding-box">
                            <strong>Chunk {i+1} → Vector</strong><br>
                            <small>[{vec_preview}, ...]</small><br>
                            <small>Dimensions: {len(vec)}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                        time.sleep(0.2)

                    rag.create_vectorstore()
                    st.session_state.ready = True
                    st.success("Vector store created!")
                    st.balloons()

# ===== TAB 2: Basic RAG =====
with tab2:
    if not api_key:
        st.warning("⚠️ Enter your Google API Key")
    elif not st.session_state.ready:
        st.warning("⚠️ Process documents first (Tab 1)")
    else:
        query = st.text_input("🔍 Your Question:", placeholder="How many days of annual leave?", key="basic_query")

        col_toggle1, col_toggle2 = st.columns(2)
        with col_toggle1:
            use_rag = st.toggle("🔌 Enable RAG", value=True)
        with col_toggle2:
            compare_mode = st.toggle("⚖️ Compare RAG vs No RAG", value=False)

        if query:
            st.markdown("---")

            if compare_mode:
                col_rag, col_no_rag = st.columns(2)

                with col_rag:
                    st.markdown("### ✅ WITH RAG")
                    with st.spinner("Processing with RAG..."):
                        results = rag.retrieve(query, k=3)
                        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
                        answer_rag = rag.generate_answer(query, context)

                    st.markdown(f'''
                    <div class="answer-box">
                        <strong>Answer:</strong><br><br>{answer_rag}
                    </div>
                    ''', unsafe_allow_html=True)

                    with st.expander("Retrieved Context"):
                        for i, (doc, score) in enumerate(results):
                            st.markdown(f"**#{i+1}** ({doc.metadata['source']}) - Distance: {score:.4f}")
                            st.text(doc.page_content[:200] + "...")

                with col_no_rag:
                    st.markdown("### ❌ WITHOUT RAG")
                    with st.spinner("Processing without RAG..."):
                        answer_no_rag = rag.generate_without_rag(query)

                    st.markdown(f'''
                    <div class="answer-box" style="border-left-color: #dc3545;">
                        <strong>Answer:</strong><br><br>{answer_no_rag}
                    </div>
                    ''', unsafe_allow_html=True)

                    st.info("No context provided - LLM uses only its general knowledge")

            else:
                if use_rag:
                    st.markdown("### 🔤 STEP 4a: Query → Embedding")
                    query_vec = rag.get_embedding(query)
                    vec_preview = ", ".join([f"{v:.3f}" for v in query_vec[:8]])
                    st.markdown(f'''
                    <div class="embedding-box">
                        <strong>Query:</strong> "{query}"<br>
                        <small>Vector: [{vec_preview}, ...]</small>
                    </div>
                    ''', unsafe_allow_html=True)

                    st.markdown("### 🎯 STEP 4b: Similarity Search")
                    results = rag.retrieve(query, k=3)
                    context_parts = []

                    for i, (doc, score) in enumerate(results):
                        context_parts.append(doc.page_content)
                        st.markdown(f'''
                        <div class="retrieval-box">
                            <strong>#{i+1}</strong>
                            <span class="score-badge">Distance: {score:.4f}</span>
                            <span class="score-badge">{doc.metadata["source"]}</span>
                            <hr>{doc.page_content}
                        </div>
                        ''', unsafe_allow_html=True)

                    st.markdown("### 📦 STEP 4c: Build Context")
                    context = "\n\n---\n\n".join(context_parts)
                    with st.expander("View Context"):
                        st.text(context)

                    st.markdown("### 🤖 STEP 4d: Generate Answer (WITH RAG)")
                    with st.spinner("Generating..."):
                        answer = rag.generate_answer(query, context)

                    st.markdown(f'''
                    <div class="answer-box">
                        <strong>Answer:</strong><br><br>{answer}
                    </div>
                    ''', unsafe_allow_html=True)

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks Retrieved", len(results))
                    col2.metric("Context Size", f"{len(context):,} chars")
                    col3.metric("Answer Size", f"{len(answer):,} chars")

                else:
                    st.markdown("### 🤖 Direct LLM (NO RAG)")
                    st.warning("⚠️ RAG is OFF - Using only LLM's general knowledge")

                    with st.spinner("Generating without context..."):
                        answer = rag.generate_without_rag(query)

                    st.markdown(f'''
                    <div class="answer-box" style="border-left-color: #dc3545;">
                        <strong>Answer:</strong><br><br>{answer}
                    </div>
                    ''', unsafe_allow_html=True)

                    st.info("💡 Notice: The LLM doesn't know about your company's specific policies!")

# ===== TAB 3: Agentic RAG =====
with tab3:
    st.markdown("""
    ### 🤖 Agentic RAG

    **What's different?** The agent *decides* whether to use RAG or not!

    - For company policy questions → Agent uses the `search_company_docs` tool
    - For general questions (math, facts) → Agent answers directly
    """)

    if not api_key:
        st.warning("⚠️ Enter your Google API Key")
    elif not st.session_state.ready:
        st.warning("⚠️ Process documents first (Tab 1)")
    else:
        agent_query = st.text_input("🤖 Ask the Agent:", placeholder="Try: 'How many leave days?' or 'What is 2+2?'", key="agent_query")

        compare_agent = st.toggle("⚖️ Compare: Agentic RAG vs Basic RAG vs No RAG", value=False, key="compare_agent")

        if agent_query:
            st.markdown("---")

            if compare_agent:
                # Three-way comparison
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 🤖 AGENTIC RAG")
                    st.caption("Agent decides when to use tools")
                    with st.spinner("Agent thinking..."):
                        try:
                            agent_result = rag.run_agent(agent_query)
                            agent_answer = agent_result["answer"]
                            tool_calls = agent_result["tool_calls"]
                        except Exception as e:
                            agent_answer = f"Error: {e}"
                            tool_calls = []

                    if tool_calls:
                        st.markdown(f'''
                        <div class="tool-call-box">
                            <strong>🔧 Tool Called:</strong> search_company_docs<br>
                            <small>Query: "{tool_calls[0].get('args', {}).get('query', 'N/A')}"</small>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ No tool used - answered directly")

                    st.markdown(f'''
                    <div class="agent-box">
                        <strong>Answer:</strong><br><br>{agent_answer}
                    </div>
                    ''', unsafe_allow_html=True)

                with col2:
                    st.markdown("### 📚 BASIC RAG")
                    st.caption("Always retrieves context")
                    with st.spinner("Processing..."):
                        results = rag.retrieve(agent_query, k=3)
                        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
                        basic_answer = rag.generate_answer(agent_query, context)

                    st.info(f"📄 Retrieved {len(results)} chunks")

                    st.markdown(f'''
                    <div class="answer-box">
                        <strong>Answer:</strong><br><br>{basic_answer}
                    </div>
                    ''', unsafe_allow_html=True)

                with col3:
                    st.markdown("### ❌ NO RAG")
                    st.caption("Pure LLM, no context")
                    with st.spinner("Processing..."):
                        no_rag_answer = rag.generate_without_rag(agent_query)

                    st.info("🧠 Using general knowledge only")

                    st.markdown(f'''
                    <div class="answer-box" style="border-left-color: #dc3545;">
                        <strong>Answer:</strong><br><br>{no_rag_answer}
                    </div>
                    ''', unsafe_allow_html=True)

            else:
                # Single agentic mode with step visualization
                st.markdown("### 🧠 Agent Reasoning")

                with st.spinner("Agent is thinking..."):
                    try:
                        agent_result = rag.run_agent(agent_query)
                    except Exception as e:
                        st.error(f"Agent error: {e}")
                        agent_result = None

                if agent_result:
                    # Show tool calls if any
                    tool_calls = agent_result["tool_calls"]
                    if tool_calls:
                        st.markdown("### 🔧 Tool Call Detected")
                        for tc in tool_calls:
                            tool_name = tc.get('name', 'unknown')
                            tool_args = tc.get('args', {})
                            st.markdown(f'''
                            <div class="tool-call-box">
                                <strong>🔧 Agent decided to use tool:</strong> {tool_name}<br>
                                <strong>Search Query:</strong> "{tool_args.get('query', 'N/A')}"
                            </div>
                            ''', unsafe_allow_html=True)

                        # Show retrieved docs
                        if agent_result["retrieved_docs"]:
                            st.markdown("### 📄 Retrieved Documents")
                            for i, doc in enumerate(agent_result["retrieved_docs"][:3]):
                                st.markdown(f'''
                                <div class="retrieval-box">
                                    <strong>#{i+1}</strong>
                                    <span class="score-badge">{doc.metadata.get("source", "N/A")}</span>
                                    <hr>{doc.page_content[:300]}...
                                </div>
                                ''', unsafe_allow_html=True)
                    else:
                        st.markdown("### 💭 Direct Response")
                        st.info("Agent decided NO tool was needed - answering from general knowledge")

                    # Final answer
                    st.markdown("### 💬 Final Answer")
                    st.markdown(f'''
                    <div class="agent-box">
                        <strong>Answer:</strong><br><br>{agent_result["answer"]}
                    </div>
                    ''', unsafe_allow_html=True)

                    # Summary
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    col1.metric("Tools Used", len(tool_calls) if tool_calls else 0)
                    col2.metric("Docs Retrieved", len(agent_result["retrieved_docs"]) if agent_result["retrieved_docs"] else 0)
