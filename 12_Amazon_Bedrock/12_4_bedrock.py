# pip install boto3
import boto3
import json
import re

# --------------------------
# AWS Bedrock Setup
# --------------------------
client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "us.amazon.nova-2-lite-v1:0"

# --------------------------
# Chat history (Nova format)
# --------------------------
chat_history = []

# --------------------------
# Guardrails
# --------------------------
def check_guardrails(user_input: str) -> str | None:
    """
    Check input against basic guardrails.
    Return an error message if input is blocked, else None.
    """
    blocked_words = ["hack", "terrorist", "bomb", "kill"]
    max_length = 300

    for word in blocked_words:
        if re.search(rf"\b{word}\b", user_input.lower()):
            return f"Sorry, I cannot discuss topics related to '{word}'."

    if len(user_input) > max_length:
        return "Your input is too long. Please shorten your question."

    return None


# --------------------------
# Ask Bedrock (Nova)
# --------------------------
def ask_bedrock(user_input: str) -> str:
    # Guardrail check
    violation = check_guardrails(user_input)
    if violation:
        return violation

    # Keep last 5 turns (10 messages: user + assistant)
    recent_history = chat_history[-10:]

    # Add current user message
    recent_history.append({
        "role": "user",
        "content": [{"text": user_input}]
    })

    payload = {
        "messages": recent_history
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    output = json.loads(response["body"].read())
    ai_answer = output["output"]["message"]["content"][0]["text"]

    # Persist full history
    chat_history.append({
        "role": "user",
        "content": [{"text": user_input}]
    })
    chat_history.append({
        "role": "assistant",
        "content": [{"text": ai_answer}]
    })

    return ai_answer


# --------------------------
# Chatbot Loop
# --------------------------
print("Nova Chatbot with Guardrails is ready! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot ended.")
        break

    answer = ask_bedrock(user_input)
    print("AI:", answer)
