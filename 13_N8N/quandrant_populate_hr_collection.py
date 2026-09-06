# pip install qdrant-client openai python-dotenv pandas

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv(override=True)

# Initialize clients
client = QdrantClient(host="localhost", port=6333)
openai_client = OpenAI()

# CSV file path
csv_file = "hr_policy_qa_2000.csv"   # keep this file in the same folder as your script

# Read CSV
df = pd.read_csv(csv_file)

collection_name = "hr_policies"

# Create collection if it does not exist
existing_collections = [c.name for c in client.get_collections().collections]

if collection_name not in existing_collections:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

batch_size = 100

for start in range(0, len(df), batch_size):
    batch_df = df.iloc[start:start + batch_size]

    questions = batch_df["question"].astype(str).tolist()

    # Get embeddings in batch
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=questions
    )

    points = []

    for j, row in enumerate(batch_df.itertuples(index=True)):
        vector = response.data[j].embedding

        points.append(
            PointStruct(
                id=int(row.Index),
                vector=vector,
                payload={
                    "question": str(row.question),
                    "answer": str(row.answer),
                    "category": str(row.category) if hasattr(row, "category") else "General"
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"Inserted records {start} to {start + len(batch_df) - 1}")

print(f"All {len(df)} Q&A records inserted into Qdrant collection '{collection_name}'!")