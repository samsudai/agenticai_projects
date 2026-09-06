# pip install boto3
import boto3
import json

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

model_id = "us.amazon.nova-2-lite-v1:0"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"text": "What is Agentic AI?"}
            ]
        }
    ]
}

response = client.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload)
)

output = json.loads(response["body"].read())
print(output["output"]["message"]["content"][0]["text"])
