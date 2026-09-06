import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "us.amazon.nova-lite-v1:0"

def ask_bedrock_stream(user_input: str):
    response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=json.dumps({
            "messages": [{"role": "user", "content": [{"text": user_input}]}],
            "inferenceConfig": {"max_new_tokens": 512}
        })
    )

    print("AI: ", end="", flush=True)
    
    for event in response["body"]:
        chunk = event.get("chunk")
        if chunk:
            data = json.loads(chunk["bytes"])
            if "contentBlockDelta" in data:
                text = data["contentBlockDelta"]["delta"].get("text", "")
                if text:
                    print(text, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    print("Nova Streaming Chatbot ready! Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        ask_bedrock_stream(user_input)