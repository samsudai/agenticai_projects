# pip install boto3
import boto3
import json

# Create a Bedrock client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Nova model ID
model_id = "us.amazon.nova-2-lite-v1:0"

# Chat history in Nova format
chat_history = []

# Helper function to send prompt to Bedrock with history
def ask_bedrock(user_input: str) -> str:
    # Add user message to history
    chat_history.append({
        "role": "user",
        "content": [
            {"text": user_input}
        ]
    })

    payload = {
        "messages": chat_history
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    output = json.loads(response["body"].read())
    ai_answer = output["output"]["message"]["content"][0]["text"]

    # Add assistant response to history
    chat_history.append({
        "role": "assistant",
        "content": [
            {"text": ai_answer}
        ]
    })

    return ai_answer

# Chatbot loop
print("Nova Chatbot with memory is ready! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot ended.")
        break

    answer = ask_bedrock(user_input)
    print("Bot:", answer)
