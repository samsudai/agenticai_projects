from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Create collection (run once)
client.create_collection(
    collection_name="hr_policies",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

print("Collection created!")
