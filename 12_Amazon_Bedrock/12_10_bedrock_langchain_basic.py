from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
import boto3

AWS_REGION = "us-west-2"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = ChatBedrock(model_id="us.amazon.nova-lite-v1:0", client=bedrock)

def first_chain():
    prompt = PromptTemplate.from_template(
        "Write a short, compelling product description for: {product_name}"
    )
    chain = prompt | model

    response = chain.invoke({"product_name": "modern computer speaker"})
    print(response.content)


first_chain()