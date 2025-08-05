import streamlit as st
import os
import tiktoken  # Import the tokenizer library
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import re

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="AI Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
# Using session_state to store data across reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- CORE APPLICATION LOGIC ---

@st.cache_resource
def get_embeddings_model():
    """Returns the embedding model, cached for efficiency."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(file_bytes, file_name):
    """Loads a PDF, splits it, creates embeddings, and stores them in FAISS."""
    if not os.path.exists("data"):
        os.makedirs("data")
    temp_file_path = os.path.join("data", file_name)
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)
    
    loader = PyMuPDFLoader(temp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    os.remove(temp_file_path)
    return vector_store, chunks

def truncate_text_by_tokens(text, model="gpt-3.5-turbo", max_tokens=3000):
    """Truncates text to a maximum number of tokens."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text

def parse_llm_output(llm_response):
    """Parses the raw LLM output for the initial clause analysis."""
    parsed_output = {
        "Summary": "Could not parse summary.", "Analysis": "Could not parse analysis.",
        "Risk Score": "Unknown", "Justification": "Could not parse justification."
    }
    summary_match = re.search(r"Summary:(.*?)(Analysis:|$)", llm_response, re.DOTALL)
    analysis_match = re.search(r"Analysis:(.*?)(Risk Score:|$)", llm_response, re.DOTALL)
    score_match = re.search(r"Risk Score:(.*?)(Justification:|$)", llm_response, re.DOTALL)
    justification_match = re.search(r"Justification:(.*)", llm_response, re.DOTALL)

    if summary_match: parsed_output["Summary"] = summary_match.group(1).strip()
    if analysis_match: parsed_output["Analysis"] = analysis_match.group(1).strip()
    if score_match:
        score = score_match.group(1).strip().lower()
        if "low" in score: parsed_output["Risk Score"] = "Low"
        elif "medium" in score: parsed_output["Risk Score"] = "Medium"
        elif "high" in score: parsed_output["Risk Score"] = "High"
    if justification_match: parsed_output["Justification"] = justification_match.group(1).strip()
    return parsed_output

# --- STREAMLIT UI ---

st.title("⚖️ AI Contract Risk Analyzer & Assistant")
st.markdown("""
Upload a contract to receive a high-level summary and a clause-by-clause risk analysis. Afterwards, ask specific follow-up questions.
""")

with st.sidebar:
    st.header("Upload Your Contract")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Analyze New Contract"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.info("Powered by Ollama and LangChain. Built for victory.")

# --- Main Logic Flow ---

# 1. Handle file upload and initial processing
if uploaded_file is not None and st.session_state.vector_store is None:
    st.header(f"Analyzing: `{uploaded_file.name}`")
    with st.spinner("Processing document... This may take a moment."):
        try:
            file_bytes = uploaded_file.getvalue()
            vector_store, chunks = create_vector_store(file_bytes, uploaded_file.name)
            st.session_state.vector_store = vector_store
            st.session_state.chunks = chunks
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()

# 2. If processing is complete, create the UI tabs
if st.session_state.vector_store is not None:
    analysis_tab, qa_tab = st.tabs(["Risk Analysis", "Q&A Assistant"])

    # --- ANALYSIS TAB ---
    with analysis_tab:
        if st.session_state.analysis_results is None:
            with st.spinner("Performing AI analysis... This may take a few minutes."):
                llm = Ollama(model="contract-analyzer-model")
                
                # Run Key Information Extraction
                full_text = " ".join([chunk.page_content for chunk in st.session_state.chunks])
                limited_text = truncate_text_by_tokens(full_text)
                extraction_prompt = ChatPromptTemplate.from_template("From the contract text below, extract these details: Landlord/Company Name, Tenant/Freelancer Name, Rent/Payment Amount, Security Deposit, Start Date, End Date. If not found, state 'Not Found'.\n\nContract Text:\n{contract_text}")
                key_info = (extraction_prompt | llm | StrOutputParser()).invoke({"contract_text": limited_text})

                # Run Clause-by-Clause Analysis
                clause_analysis_prompt = ChatPromptTemplate.from_template("You are a paralegal. Analyze the following contract clause. Respond ONLY with this structure:\nSummary: [summary]\nAnalysis: [risk analysis]\nRisk Score: [Low/Medium/High]\nJustification: [justification]\n\nCONTEXT:\n{context}")
                analysis_chain = clause_analysis_prompt | llm | StrOutputParser()
                clause_results = [parse_llm_output(analysis_chain.invoke({"context": chunk.page_content})) for chunk in st.session_state.chunks]

                # Run Executive Summary
                analyses_text = "\n---\n".join([str(res) for res in clause_results])
                limited_analyses_text = truncate_text_by_tokens(analyses_text)
                summary_prompt = ChatPromptTemplate.from_template("Based on the following analyses, write a one-paragraph executive summary, mention the overall risk, and point out key clauses to review.\n\nAnalyses:\n{analyses_text}")
                final_summary = (summary_prompt | llm | StrOutputParser()).invoke({"analyses_text": limited_analyses_text})

                # Store all results in session state
                st.session_state.analysis_results = {
                    "key_info": key_info,
                    "clause_results": clause_results,
                    "final_summary": final_summary
                }
                st.rerun() # Rerun to display the results now that they are calculated
        else:
            # Display the pre-calculated results
            st.subheader("Key Information Summary")
            st.info(st.session_state.analysis_results["key_info"])

            st.subheader("Overall Executive Summary")
            st.success(st.session_state.analysis_results["final_summary"])
            
            st.subheader("Clause-by-Clause Risk Analysis:")
            for i, result in enumerate(st.session_state.analysis_results["clause_results"]):
                with st.container(border=True):
                    risk_score = result["Risk Score"]
                    if risk_score == "Low": st.success(f"**Risk Score: {risk_score}**")
                    elif risk_score == "Medium": st.warning(f"**Risk Score: {risk_score}**")
                    elif risk_score == "High": st.error(f"**Risk Score: {risk_score}**")
                    else: st.info(f"**Risk Score: {risk_score}**")
                    
                    st.markdown(f"**Summary:** {result['Summary']}")
                    st.markdown(f"**Potential Risk Analysis:** {result['Analysis']}")
                    with st.expander("Show Original Clause Text"):
                        st.text_area("", st.session_state.chunks[i].page_content, height=150, disabled=True, key=f"chunk_{i}")

    # --- Q&A TAB ---
    with qa_tab:
        st.header("Interactive Q&A Assistant")
        st.markdown("Ask follow-up questions about the contract.")

        # Display existing chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for user's question
        if prompt := st.chat_input("What is the notice period for termination?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retriever = st.session_state.vector_store.as_retriever()
                    
                    # --- UPGRADED PROMPT ---
                    rag_prompt_template = """
                    You are a helpful legal assistant. Your task is to answer the user's question based on the contract context provided.

                    1. First, carefully read the context and try to answer the question directly from it.
                    2. If the answer is clearly present, provide it.
                    3. If the answer is not in the context, state that the document does not contain this information, and then provide a general, helpful answer based on your knowledge as a legal assistant. Start this general answer with: "Based on my general knowledge..."

                    CONTEXT:
                    {context}
                    
                    QUESTION:
                    {question}
                    """
                    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
                    llm = Ollama(model="contract-analyzer-model")
                    
                    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | rag_prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Initial state if no file is uploaded
if uploaded_file is None:
    st.info("Upload a contract PDF in the sidebar to begin analysis.")
