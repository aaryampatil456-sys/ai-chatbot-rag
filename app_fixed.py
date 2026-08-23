import streamlit as st
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_utils import (
    load_and_split_pdf,
    create_vectorstore,
    search_vectorstore
)

# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main Header */
.hero {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid rgba(128,128,128,0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.75;
}

/* Cards */
.info-card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 18px;
}

.status-card {
    padding: 16px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 10px;
}

/* Section titles */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
}

/* Answer */
.answer-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-top: 10px;
}

/* Sidebar */
.sidebar-title {
    font-size: 24px;
    font-weight: 700;
}

.small-text {
    font-size: 13px;
    opacity: 0.7;
}

/* Hide default footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🤖 AI PDF Chatbot
</div>

<div class="hero-subtitle">
Intelligent document question answering using
Retrieval-Augmented Generation (RAG)
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# API KEY CHECK
# =========================================================

if not GOOGLE_API_KEY:

    st.error(
        "❌ Gemini API Key not found. "
        "Please add GOOGLE_API_KEY to your .env file."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📄 Document</div>',
        unsafe_allow_html=True
    )

    st.write("Upload a PDF and start asking questions.")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown("### ⚙️ RAG Pipeline")

    st.markdown("""
    **1. 📄 PDF Upload**

    **2. ✂️ Text Chunking**

    **3. 🧠 HuggingFace Embeddings**

    **4. 🔎 FAISS Retrieval**

    **5. 🤖 Gemini Answer**
    """)

    st.divider()

    if st.session_state.pdf_ready:

        st.success("🟢 Document Ready")

        st.caption(
            f"📘 {st.session_state.pdf_name}"
        )

        st.caption(
            f"📊 {st.session_state.chunk_count} text chunks"
        )

    else:

        st.info("🔵 No document uploaded")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# PDF PROCESSING
# =========================================================

if uploaded_file is not None:

    if (
        not st.session_state.pdf_ready
        or st.session_state.pdf_name != uploaded_file.name
    ):

        os.makedirs("uploads", exist_ok=True)

        pdf_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(
            "📚 Reading and processing your PDF..."
        ):

            try:

                chunks = load_and_split_pdf(
                    pdf_path
                )

                create_vectorstore(
                    chunks
                )

                st.session_state.pdf_ready = True

                st.session_state.pdf_name = (
                    uploaded_file.name
                )

                st.session_state.chunk_count = (
                    len(chunks)
                )

                st.session_state.messages = []

                st.success(
                    f"✅ Document processed successfully! "
                    f"{len(chunks)} text chunks created."
                )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF:\n\n{e}"
                )

                st.stop()


# =========================================================
# DOCUMENT STATUS
# =========================================================

if st.session_state.pdf_ready:

    st.markdown(
        f"""
        <div class="status-card">

        <b>🟢 Document Ready</b><br>

        📘 {st.session_state.pdf_name}

        &nbsp;&nbsp;|&nbsp;&nbsp;

        📊 {st.session_state.chunk_count} text chunks

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown("""
    <div class="info-card">

    <div class="section-title">
    👋 Welcome!
    </div>

    Upload a PDF from the sidebar to start
    chatting with your document.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# CHAT AREA
# =========================================================

st.markdown(
    '<div class="section-title">💬 Chat with your PDF</div>',
    unsafe_allow_html=True
)


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask anything about your PDF..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    if not st.session_state.pdf_ready:

        st.warning(
            "⚠️ Please upload and process a PDF first."
        )

        st.stop()


    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Searching your PDF and generating answer..."
        ):

            try:

                results = search_vectorstore(
                    question,
                    k=4
                )

                context = "\n\n".join(
                    document.page_content
                    for document in results
                )

                # Handle Unicode characters safely before sending context to the model
                context = context.encode("ascii", "ignore").decode("ascii")
                question = question.encode("ascii", "ignore").decode("ascii")


                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.6-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2
                )

                prompt = f"""
You are an intelligent AI assistant that answers
questions using the uploaded PDF.

Use the PDF context provided below.

IMPORTANT RULES:

1. Answer naturally like ChatGPT.
2. Do NOT return JSON.
3. Do NOT return Python dictionaries.
4. Do NOT show fields such as type, text, extras,
   signature, or metadata.
5. Give a clear and human-readable answer.
6. Use headings and bullet points when useful.
7. If the user asks what subject the PDF is about,
   identify the subject clearly.
8. If the answer is not available in the PDF,
   clearly say that it is not available.
9. Do not make up information.
10. Keep the answer relevant to the question.

PDF CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}

Give the final answer directly.
"""


                # Generate answer

                response = llm.invoke(
                    prompt
                )

                content = response.content

                if isinstance(content, str):

                    answer = content

                elif isinstance(content, list):

                    text_parts = []

                    for item in content:

                        if isinstance(item, dict):

                            if item.get("type") == "text":

                                text_parts.append(
                                    item.get("text", "")
                                )

                            elif "text" in item:

                                text_parts.append(
                                    str(item["text"])
                                )

                        else:

                            text_parts.append(
                                str(item)
                            )

                    answer = "\n".join(
                        text_parts
                    )

                else:

                    answer = str(content)

                st.markdown(
                    '<div class="answer-card">',
                    unsafe_allow_html=True
                )

                st.markdown(answer)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                with st.expander(
                    "📚 View information retrieved from PDF"
                ):

                    st.caption(
                        "These are the relevant sections retrieved "
                        "from your uploaded document."
                    )

                    for i, document in enumerate(
                        results
                    ):

                        st.markdown(
                            f"### 📄 Source {i + 1}"
                        )

                        if hasattr(
                            document,
                            "metadata"
                        ):

                            page = document.metadata.get(
                                "page"
                            )

                            if page is not None:

                                st.caption(
                                    f"📖 Page {page + 1}"
                                )

                        st.write(
                            document.page_content
                        )

                        st.divider()


                # Save conversation

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                st.error(
                    f"❌ Error generating answer:\n\n{e}"
                )

st.divider()

st.caption(
    "🤖 AI PDF Chatbot • "
    "Powered by RAG + FAISS + HuggingFace + Gemini"
)