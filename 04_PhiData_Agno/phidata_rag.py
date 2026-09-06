# ================================
# PhiData + RAG using ChromaDB
# Single-file working example
# ================================

# --------- INSTALL (once) ----------
# pip install phidata chromadb sentence-transformers openai

# --------- IMPORTS ----------
import chromadb
from chromadb.utils import embedding_functions

from phi.agent import Agent
from phi.model.openai import OpenAIChat

from dotenv import load_dotenv
load_dotenv(override=True)

# --------- STEP 1: DOCUMENTS ----------
documents = [
    "Docker is a containerization platform used to package applications.",
    "Kubernetes is a container orchestration system for managing containers.",
    "Terraform is an Infrastructure as Code tool used to provision resources.",
    "Ansible is a configuration management and automation tool.",
    "GitHub Actions is a CI/CD tool integrated into GitHub."
]

# --------- STEP 2: CREATE CHROMADB ----------
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()

collection = client.create_collection(
    name="devops_rag",
    embedding_function=embedding_function
)

collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))]
)

# --------- STEP 3: RETRIEVER TOOL ----------
def retrieve_docs(query: str) -> str:
    """
    Retrieve relevant documents from ChromaDB
    """
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    return "\n".join(results["documents"][0])

# --------- STEP 4: PHIDATA AGENT ----------
agent = Agent(
    model=OpenAIChat(model="gpt-4o-mini"),
    tools=[retrieve_docs],
    instructions="""
    You are a DevOps assistant.
    Use retrieved documents to answer questions.
    Do not hallucinate.
    """
)

# --------- STEP 5: ASK QUESTIONS ----------
agent.print_response("What is Terraform?")
agent.print_response("Explain Docker and Kubernetes")
agent.print_response("Which tool is used for CI/CD?")
