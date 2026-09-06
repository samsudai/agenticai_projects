# ---------------------------------------------------------
# policy_loader.py
# ---------------------------------------------------------
# This file loads the customer support policy PDF,
# splits it into chunks, creates a FAISS vector database,
# and searches the most relevant policy sections.
# ---------------------------------------------------------

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Recommended import for newer LangChain versions.
# If this does not work, check that langchain-huggingface is installed.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# File Paths
# -----------------------------
POLICY_PDF_PATH = Path("policy_knowledge.pdf")
FAISS_INDEX_PATH = Path("vector_store/policy_faiss_index")


# -----------------------------
# Embedding Model
# -----------------------------
def get_embeddings():
    """
    Creates the embedding model used for semantic search.

    This model converts policy text and user queries into vectors
    so that similar content can be searched.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# Load PDF
# -----------------------------
def load_policy_pdf():
    """
    Loads the customer support policy PDF.

    Returns:
        List of LangChain document objects.
    """

    if not POLICY_PDF_PATH.exists():
        raise FileNotFoundError(
            f"Policy PDF not found at: {POLICY_PDF_PATH}. "
            "Please add customer_support_policy.pdf."
        )

    loader = PyPDFLoader(str(POLICY_PDF_PATH))
    documents = loader.load()

    return documents


# -----------------------------
# Split PDF into Chunks
# -----------------------------
def split_policy_documents(documents):
    """
    Splits the PDF content into smaller chunks.

    Why?
    The AI agent should not read the entire PDF every time.
    Smaller chunks make search faster and more accurate.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# -----------------------------
# Build FAISS Vector Store
# -----------------------------
def build_policy_vector_store():
    """
    Builds a FAISS vector store from the policy PDF
    and saves it locally.

    Run automatically if the vector store does not already exist.
    """

    documents = load_policy_pdf()
    chunks = split_policy_documents(documents)
    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_INDEX_PATH))

    return vector_store


# -----------------------------
# Load Existing FAISS Vector Store
# -----------------------------
def load_policy_vector_store():
    """
    Loads the saved FAISS vector store.

    If the FAISS index does not exist yet, it creates one from the PDF.
    """

    embeddings = get_embeddings()

    if not FAISS_INDEX_PATH.exists():
        return build_policy_vector_store()

    vector_store = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


# -----------------------------
# Search Policy PDF
# -----------------------------
def search_policy(query: str, top_k: int = 3) -> str:
    """
    Searches the policy PDF and returns the most relevant policy text.

    Args:
        query: Customer issue or issue category.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        Relevant policy text as a string.
    """

    if not query:
        query = "general customer support policy"

    vector_store = load_policy_vector_store()

    relevant_docs = vector_store.similarity_search(
        query=query,
        k=top_k
    )

    if not relevant_docs:
        return "No relevant policy section found in the PDF."

    policy_sections = []

    for index, doc in enumerate(relevant_docs, start=1):
        page_number = doc.metadata.get("page", "Unknown")

        section_text = f"""
--- Policy Section {index} ---
Source Page: {page_number}
{doc.page_content}
"""
        policy_sections.append(section_text)

    return "\n".join(policy_sections)


# -----------------------------
# Function Used by the Agent
# -----------------------------
def get_relevant_policy(issue_category: str, customer_message: str) -> str:
    """
    Creates a better search query using both:
    1. Issue category
    2. Customer message

    This helps the agent retrieve the most useful policy section.
    """

    query = f"""
Issue category: {issue_category}

Customer message:
{customer_message}

Find the most relevant customer support policy, rules, escalation guidance,
refund rules, delivery rules, cancellation rules, billing rules, or warranty rules.
"""

    return search_policy(query=query, top_k=3)