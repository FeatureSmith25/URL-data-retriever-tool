import streamlit as st
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from rich import print

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2603")
embedding = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

st.set_page_config(
    page_title="Website Chat",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp{
    background:#0f0f0f;
}

/* Chat input container */
[data-testid="stChatInput"]{
    position:fixed;
    bottom:25px;
    left:50%;
    transform:translateX(-50%);
    width:75%;
    z-index:999;
}

/* Rounded input */
[data-testid="stChatInput"] textarea{
    background:#2b2b2b !important;
    color:white !important;
    border:none !important;
    border-radius:30px !important;
    padding:18px 60px 18px 18px !important;
    font-size:18px !important;
    min-height:55px !important;
    box-shadow:none !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder{
    color:#a0a0a0;
}

/* Remove red border */
[data-testid="stChatInput"] textarea:focus{
    border:none !important;
    box-shadow:none !important;
    outline:none !important;
}

/* Send button */
[data-testid="stChatInput"] button{
    background:#3b82f6 !important;
    border-radius:50% !important;
    width:38px;
    height:38px;
    border:none !important;
}

[data-testid="stChatInput"] button:hover{
    background:#2563eb !important;
}

</style>
""", unsafe_allow_html=True)

st.title("🌐 Website Chat")

# with st.sidebar:
#     st.header("Settings")

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "retriever" not in st.session_state:
    st.session_state.retriever=None

if "website" not in st.session_state:
    st.session_state.website=None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt=st.chat_input(
    "Paste website URL or ask a question..."
)

if user_prompt:

    st.session_state.messages.append(
        {"role":"user","content":user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    url_pattern=r'https?://\S+'

    if re.match(url_pattern,user_prompt):

        with st.spinner("Loading website..."):

            docs=WebBaseLoader(user_prompt).load()

            splitter=RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            split_docs=splitter.split_documents(docs)

            vectordb=Chroma.from_documents(
                documents=split_docs,
                embedding=embedding
            )

            st.session_state.retriever=vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k":1}
            )

            st.session_state.website=user_prompt

        reply=f"""
✅ Website loaded successfully!

**{user_prompt}**

Now ask me anything about this website.
"""

    else:

        if st.session_state.retriever is None:

            reply="Please paste a website URL first."

        else:

            docs=st.session_state.retriever.invoke(user_prompt)

            context="\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are an expert AI assistant.

Answer the user's question ONLY from the provided context.
 
Instructions:
- Try to give brief out but if don't have enough data then provide amount of data website contain.
- Explain the topic clearly.
- Include all important points from the context.
- Use headings and bullet points whenever appropriate.
- If there are steps, write them in numbered form.
- If examples are available in the context, include them.
- Do not make up information outside the context.
- If the context does not contain enough information then give containing data.
  

Context:
{context}

Question:
{user_prompt}

Detailed Answer:
"""

            response=llm.invoke(prompt)

            parser=StrOutputParser()

            reply=parser.invoke(response)

    st.session_state.messages.append(
        {"role":"assistant","content":reply}
    )

    with st.chat_message("assistant"):
        st.markdown(reply)