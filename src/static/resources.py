import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

@st.cache_resource
def build_prompt(system_prompt_str: str):
    """Construct a ChatPromptTemplate from a system prompt string"""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt_str.strip() +
            "\n\nYour task is to provide a clear, informative, and beautifully formatted markdown response for each topic provided." +
            "\nInclude headings, bullet points, code blocks, and other markdown elements where appropriate." +
            "\n\nContext:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


@st.cache_resource
def create_chains(system_prompt_str: str):
    """Create the document chain using a system prompt string"""
    prompt_template = build_prompt(system_prompt_str)
    return create_stuff_documents_chain(get_llm(), prompt_template)
