from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import boto3

my_data = [
    "Mumbai is the financial capital of India.",
    "The festival of Pongal is widely celebrated in Tamil Nadu.",
    "ISRO launched the Chandrayaan-3 mission in 2023.",
    "Rohit Sharma is one of India's most successful cricket captains."
]

question = "Which Indian state celebrates Pongal?"

AWS_REGION = "us-west-2"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = ChatBedrock(model_id="us.amazon.nova-lite-v1:0", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0", client=bedrock
)

vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

results = retriever.invoke(question)

results_string = []
for result in results:
    results_string.append(result.page_content)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}"),
    ]
)

chain = template | model

response = chain.invoke({"input": question, "context": results_string})
print(response.content)