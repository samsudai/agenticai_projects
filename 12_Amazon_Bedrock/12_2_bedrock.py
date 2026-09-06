# pip install boto3
import boto3
import json

# Create a Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Nova model ID
model_id = "us.amazon.nova-2-lite-v1:0"

# Helper function to send prompt to Bedrock (Nova)
def ask_bedrock(prompt: str) -> str:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
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

    # Nova response parsing
    return output["output"]["message"]["content"][0]["text"]

# Simple chatbot loop
print("Nova Chatbot is ready! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot ended.")
        break

    answer = ask_bedrock(user_input)
    print("Bot:", answer)
