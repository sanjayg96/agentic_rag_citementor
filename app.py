import streamlit as st
import time
import pandas as pd
from rag_engine import RAGEngine

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CiteMentor | Attribution-Aware AI",
    page_icon="⚖️",
    layout="wide"
)

# Initialize Page State
if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
    <style>
    /* Headers */
    .main-header {
        font-size: 3em;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: -10px;
    }
    .sub-header {
        font-size: 1.2em;
        color: #e0e0e0 !important;
        font-style: italic;
        margin-bottom: 20px;
    }
    
    /* Source Card Styling */
    .source-citation {
        background-color: #262730;
        border-left: 4px solid #34d399; /* Green accent */
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
    .source-title {
        font-weight: bold;
        font-size: 0.95em;
        color: #ffffff;
    }
    .source-author {
        font-size: 0.8em;
        color: #aaaaaa;
        margin-bottom: 5px;
    }
    .source-preview {
        font-style: italic;
        font-size: 0.85em;
        color: #d1d5db; /* Light gray for readability */
        margin: 5px 0;
        line-height: 1.4;
    }
    .source-cost {
        font-size: 0.85em;
        color: #34d399;
        font-weight: bold;
        text-align: right;
    }
    
    /* Nav Button */
    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. INITIALIZE ENGINE
# ==========================================
@st.cache_resource
def get_engine():
    return RAGEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to load RAG Engine. Check your .env file and API keys.\nError: {e}")
    st.stop()

# ==========================================
# 4. SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0

# ==========================================
# 5. HEADER & NAVIGATION
# ==========================================
col_header, col_nav = st.columns([6, 1])

with col_header:
    st.markdown('<div class="main-header">CiteMentor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">The Attribution-Aware Knowledge Engine</div>', unsafe_allow_html=True)

with col_nav:
    st.write("") 
    st.write("") 
    if st.session_state.current_view == "chat":
        if st.button("📜Details", help="Learn more about this project"):
            st.session_state.current_view = "mission"
            st.rerun()
    else:
        if st.button("💬 Back to Chat"):
            st.session_state.current_view = "chat"
            st.rerun()

st.divider()

# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 💎 Micro-Royalty Ledger")
    st.info("**Fair-Use Accounting:**\nYou don't buy books here. You only pay for the specific paragraphs used to answer your question.")
    
    st.markdown(f"""
        <div style="text-align: center; padding: 15px; background-color: #0e1117; border-radius: 10px; border: 1px solid #34d399;">
            <div style="font-size: 0.9em; color: #34d399;">Current Session Total</div>
            <div style="font-size: 2.2em; font-weight: bold; color: #ffffff;">${st.session_state.session_cost:.5f}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.rerun()

# ==========================================
# 7. MAIN LOGIC
# ==========================================

# Helper function to render sources
def render_sources(sources, genre, total_cost):
    with st.expander(f"📊 Attribution & Cost (Total: ${total_cost:.5f})"):
        st.caption(f"**Genre Detected:** `{genre}`")
        for source in sources:
            st.markdown(f"""
            <div class="source-citation">
                <div class="source-title">{source['book']}</div>
                <div class="source-author">by {source['author']}</div>
                <div class="source-preview">"{source['preview']}"</div>
                <div class="source-cost">Micro-Cost: ${source['cost']:.5f}</div>
            </div>
            """, unsafe_allow_html=True)

if st.session_state.current_view == "chat":
    # --- CHAT VIEW ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                meta = message["metadata"]
                render_sources(meta['sources'], meta['genre'], meta['cost'])

    if prompt := st.chat_input("Is my destiny determined by my thoughts?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.status("Thinking...", expanded=True)
            message_placeholder = st.empty()
            
            try:
                # A. Routing
                status_placeholder.write("🧠 Analyzing intent & Routing to correct library...")
                genre = engine.route_query(prompt)
                status_placeholder.write(f"📂 Accessing Shelf: **{genre.upper()}**")
                
                # B. Retrieval
                status_placeholder.write("🔍 Searching vector database & Calculating micro-royalties...")
                result = engine.generate_answer(prompt, genre)
                
                status_placeholder.update(label="Answer Ready!", state="complete", expanded=False)
                
                # C. Display Answer
                message_placeholder.markdown(result['answer'])
                
                # D. Render Sources
                render_sources(result['sources'], result['genre'], result['total_cost'])

                # State Update
                st.session_state.session_cost += result['total_cost']
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['answer'],
                    "metadata": {
                        "cost": result['total_cost'],
                        "sources": result['sources'],
                        "genre": result['genre']
                    }
                })
                time.sleep(0.1)
                st.rerun()
                
            except Exception as e:
                status_placeholder.update(label="Error", state="error")
                st.error(f"An error occurred: {e}")

else:
    # --- MISSION VIEW ---
    st.markdown("### A Retrieval-Augmented Generation (RAG) System for Attribution-Aware Advice")
    st.info("💡 **Project Goal:** To transform static libraries into interactive mentorship while solving the AI copyright dilemma via micro-royalties.")

    st.subheader("1. The Problem: The 'Black Box' of AI")
    col1, col2 = st.columns(2)
    with col1:
        st.warning("**For Readers:**")
        st.write("Generic AI provides unverified advice. There is no 'Source of Truth'.")
    with col2:
        st.error("**For Authors:**")
        st.write("Intellectual property is used to train models without consent or compensation.")

    st.subheader("2. The Solution: CiteMentor")
    st.success("**An AI-assisted 'Mentor' that sits on top of a curated library of high-quality non-fiction.**")
    
    st.markdown("""
    * **Context-Aware Advisory:** The system retrieves specific strategies from verified experts.
    * **Verifiable Trust:** Every answer is accompanied by a "Source Card" displaying the exact book and cost.
    * **The 'Fair-Use' Ledger:** A built-in accounting mechanism that tracks content usage per query.
    """)

    st.subheader("3. Core Value Proposition")
    data = {
        "Feature": ["Curated RAG", "Citation Engine", "Royalty Logic"],
        "Benefit to User": [
            "High-quality, specific advice. No hallucinations.",
            "Trust and transparency.",
            "Users feel they are supporting creators."
        ],
        "Benefit to Ecosystem": [
            "Eliminates 'noise' from internet-scraped data.",
            "Drives discovery of the original books.",
            "Solves the ethical dilemma of AI vs. Copyright."
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)
    
    st.markdown("""
    * The system also addresses the friction point between "buying a book" and "getting value from a book."
    """)
    
    st.markdown("---")
    if st.button("✨ Try the Demo Now"):
        st.session_state.current_view = "chat"
        st.rerun()



#####################################
# import streamlit as st
# import time
# import pandas as pd
# from rag_engine import RAGEngine

# # ==========================================
# # 1. PAGE CONFIGURATION
# # ==========================================
# st.set_page_config(
#     page_title="CiteMentor | Attribution-Aware AI",
#     page_icon="⚖️",
#     layout="wide"
# )

# # Initialize Page State (Default to Chat)
# if "current_view" not in st.session_state:
#     st.session_state.current_view = "chat"

# # ==========================================
# # 2. CUSTOM CSS (Bright White Headers)
# # ==========================================
# st.markdown("""
#     <style>
#     /* Force text colors to white for Dark Mode */
#     .main-header {
#         font-size: 3em;
#         font-weight: 800;
#         color: #ffffff !important;
#         margin-bottom: -10px;
#     }
#     .sub-header {
#         font-size: 1.2em;
#         color: #e0e0e0 !important; /* Slightly off-white for subtitle */
#         font-style: italic;
#         margin-bottom: 20px;
#     }
    
#     /* Styling for the Source/Cost Cards */
#     .source-citation {
#         font-size: 0.9em;
#         color: #cccccc;
#         border-left: 3px solid #00d26a; /* Green accent */
#         padding-left: 10px;
#         margin-top: 5px;
#         background-color: #262730;
#         padding: 10px;
#         border-radius: 0 5px 5px 0;
#     }
    
#     /* Button Styling to look like a Nav Link */
#     div.stButton > button {
#         width: 100%;
#         border-radius: 20px;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 3. INITIALIZE ENGINE (Cached)
# # ==========================================
# @st.cache_resource
# def get_engine():
#     return RAGEngine()

# try:
#     engine = get_engine()
# except Exception as e:
#     st.error(f"Failed to load RAG Engine. Check your .env file and API keys.\nError: {e}")
#     st.stop()

# # ==========================================
# # 4. SESSION STATE (History & Costs)
# # ==========================================
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "session_cost" not in st.session_state:
#     st.session_state.session_cost = 0.0

# # ==========================================
# # 5. HEADER & NAVIGATION (Top Layout)
# # ==========================================

# # Create two columns: Title on Left, Nav Button on Right
# col_header, col_nav = st.columns([6, 1])

# with col_header:
#     st.markdown('<div class="main-header">CiteMentor</div>', unsafe_allow_html=True)
#     st.markdown('<div class="sub-header">The Attribution-Aware Knowledge Engine</div>', unsafe_allow_html=True)

# with col_nav:
#     st.write("") # Spacer to align button
#     st.write("") 
#     # Logic to toggle the button text and view
#     if st.session_state.current_view == "chat":
#         if st.button("📜 About this Project"):
#             st.session_state.current_view = "mission"
#             st.rerun()
#     else:
#         if st.button("💬 Back to Chat"):
#             st.session_state.current_view = "chat"
#             st.rerun()

# st.divider()

# # ==========================================
# # 6. SIDEBAR (Persistent Metrics)
# # ==========================================
# with st.sidebar:
#     st.markdown("### 💎 Micro-Royalty Ledger")
#     st.info(
#         "**Fair-Use Accounting:**\nYou don't buy books here. You only pay for the specific paragraphs used to answer your question."
#     )
    
#     # Large Cost Display
#     st.markdown(f"""
#         <div style="text-align: center; padding: 15px; background-color: #0e1117; border-radius: 10px; border: 1px solid #34d399;">
#             <div style="font-size: 0.9em; color: #34d399;">Current Session Total</div>
#             <div style="font-size: 2.2em; font-weight: bold; color: #ffffff;">${st.session_state.session_cost:.5f}</div>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("---")
#     if st.button("Reset Session"):
#         st.session_state.messages = []
#         st.session_state.session_cost = 0.0
#         st.rerun()

# # ==========================================
# # 7. PAGE CONTENT LOGIC
# # ==========================================

# if st.session_state.current_view == "chat":
#     # --- CHAT VIEW ---
    
#     # Display History
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if "metadata" in message:
#                 meta = message["metadata"]
#                 with st.expander(f"📊 Attribution & Cost (Total: ${meta['cost']:.5f})"):
#                     st.caption(f"**Genre Detected:** `{meta['genre']}`")
#                     for source in meta['sources']:
#                         st.markdown(f"""
#                         <div class="source-citation">
#                             <b>{source['book']}</b> by {source['author']}<br>
#                             Micro-Cost: ${source['cost']:.5f}
#                         </div>
#                         """, unsafe_allow_html=True)

#     # Chat Input
#     if prompt := st.chat_input("How do I handle difficult people?"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             status_placeholder = st.status("Thinking...", expanded=True)
#             message_placeholder = st.empty()
            
#             try:
#                 # A. Routing
#                 status_placeholder.write("🧠 Analyzing intent & Routing to correct library...")
#                 genre = engine.route_query(prompt)
#                 status_placeholder.write(f"📂 Accessing Shelf: **{genre.upper()}**")
                
#                 # B. Retrieval
#                 status_placeholder.write("🔍 Searching vector database & Calculating micro-royalties...")
#                 result = engine.generate_answer(prompt, genre)
                
#                 status_placeholder.update(label="Answer Ready!", state="complete", expanded=False)
                
#                 # C. Display Answer
#                 message_placeholder.markdown(result['answer'])
                
#                 # D. Metadata
#                 with st.expander(f"📊 Attribution & Cost (Total: ${result['total_cost']:.5f})"):
#                     st.caption(f"**Genre Detected:** `{result['genre']}`")
#                     for source in result['sources']:
#                         st.markdown(f"""
#                         <div class="source-citation">
#                             <b>{source['book']}</b> by {source['author']}<br>
#                             Micro-Cost: ${source['cost']:.5f}
#                         </div>
#                         """, unsafe_allow_html=True)

#                 # State Update
#                 st.session_state.session_cost += result['total_cost']
#                 st.session_state.messages.append({
#                     "role": "assistant", 
#                     "content": result['answer'],
#                     "metadata": {
#                         "cost": result['total_cost'],
#                         "sources": result['sources'],
#                         "genre": result['genre']
#                     }
#                 })
#                 time.sleep(0.1)
#                 st.rerun()
                
#             except Exception as e:
#                 status_placeholder.update(label="Error", state="error")
#                 st.error(f"An error occurred: {e}")

# else:
#     # --- MISSION & VISION VIEW ---
#     st.markdown("### A Retrieval-Augmented Generation (RAG) System for Attribution-Aware Advice")
    
#     st.info("💡 **Project Goal:** To transform static libraries into interactive mentorship while solving the AI copyright dilemma via micro-royalties.")

#     # 1. The Problem
#     st.subheader("1. The Problem: The 'Black Box' of AI")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.warning("**For Readers:**")
#         st.write("Generic AI provides unverified advice. There is no 'Source of Truth'.")
#     with col2:
#         st.error("**For Authors:**")
#         st.write("Intellectual property is used to train models without consent or compensation.")

#     # 2. The Solution
#     st.subheader("2. The Solution: CiteMentor")
#     st.success("**An AI-assisted 'Mentor' that sits on top of a curated library of high-quality non-fiction.**")
    
#     st.markdown("""
#     * **Context-Aware Advisory:** The system retrieves specific strategies from verified experts.
#     * **Verifiable Trust:** Every answer is accompanied by a "Source Card" displaying the exact book and cost.
#     * **The 'Fair-Use' Ledger:** A built-in accounting mechanism that tracks content usage per query.
#     """)

#     # 3. Value Table
#     st.subheader("3. Core Value Proposition")
    
#     data = {
#         "Feature": ["Curated RAG", "Citation Engine", "Royalty Logic"],
#         "Benefit to User": [
#             "High-quality, specific advice. No hallucinations.",
#             "Trust and transparency. The user sees the evidence.",
#             "Users feel they are supporting creators."
#         ],
#         "Benefit to Ecosystem": [
#             "Eliminates 'noise' from internet-scraped data.",
#             "Drives discovery of the original books.",
#             "Solves the ethical dilemma of AI vs. Copyright."
#         ]
#     }
#     df = pd.DataFrame(data)
#     st.table(df)
    
#     # Bottom call to action
#     st.markdown("---")
#     if st.button("✨ Try the Demo Now"):
#         st.session_state.current_view = "chat"
#         st.rerun()