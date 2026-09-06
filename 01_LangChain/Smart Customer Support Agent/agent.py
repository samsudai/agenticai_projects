import os
import json
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from order_data import get_order_details
from policy_loader import get_relevant_policy


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Create Groq LLM
# ---------------------------------------------------------
def get_llm():
    """
    Creates the Groq LLM connection.
    """

    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please add your Groq API key in the .env file."
        )

    return ChatGroq(
        model=model_name,
        temperature=0.2
    )


# ---------------------------------------------------------
# Safely extract JSON from LLM response
# ---------------------------------------------------------
def extract_json(text):
    """
    Extracts JSON from the LLM response.
    This is useful because sometimes the LLM may return extra text.
    """

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "issue_category": "Other",
        "urgency": "Medium",
        "sentiment": "Neutral",
        "summary": "Could not clearly classify the issue.",
        "missing_information": ["More details may be required."],
        "needs_human_escalation": True
    }


# ---------------------------------------------------------
# Prompt 1: Classification Prompt
# ---------------------------------------------------------
CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a customer support triage agent.

Your task is to classify the customer issue.

Return ONLY valid JSON.
Do not include markdown.
Do not include any explanation outside JSON.

Allowed issue categories:
- Delivery Delay
- Missing Package
- Refund Delay
- Product Defect
- Cancellation
- Billing Issue
- Warranty Support
- General Complaint
- Other

Allowed urgency values:
- Low
- Medium
- High

Allowed sentiment values:
- Positive
- Neutral
- Negative
- Angry

Return JSON in this exact structure:
{{
  "issue_category": "one category from the allowed list",
  "urgency": "Low, Medium, or High",
  "sentiment": "Positive, Neutral, Negative, or Angry",
  "summary": "short summary of the customer issue",
  "missing_information": ["list of missing details needed from the customer"],
  "needs_human_escalation": true or false
}}
"""
        ),
        (
            "human",
            """
Customer name: {customer_name}
Customer type: {customer_type}
Order ID: {order_id}

Customer message:
{customer_message}
"""
        )
    ]
)


# ---------------------------------------------------------
# Prompt 2: Final Response Prompt
# ---------------------------------------------------------
FINAL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a professional customer support AI assistant.

You support human customer service agents by:
1. Understanding the customer issue
2. Checking order details
3. Referring to the support policy
4. Deciding the next action
5. Drafting a polite customer response

Use only the information provided.
Do not invent facts.
Do not promise refund, replacement, cancellation, or compensation unless the policy clearly supports it.

Return the answer in this exact format:

### Issue Category
...

### Urgency Level
...

### Sentiment
...

### Case Summary
...

### Order Check
...

### Relevant Policy
...

### Recommended Action
...

### Information Needed
...

### Customer Response Draft
...

### Internal Note for Support Agent
...
"""
        ),
        (
            "human",
            """
Customer name: {customer_name}
Customer type: {customer_type}
Order ID: {order_id}

Customer message:
{customer_message}

Issue classification:
{classification}

Order details:
{order_details}

Relevant policy from PDF:
{policy_text}
"""
        )
    ]
)


# ---------------------------------------------------------
# Main Agent Workflow
# ---------------------------------------------------------
def run_support_agent(customer_message, customer_name, customer_type, order_id):
    """
    Main customer support agent workflow.

    Steps:
    1. Classify the customer issue.
    2. Read order details from CSV.
    3. Retrieve relevant policy from PDF.
    4. Generate a structured customer support response.
    """

    llm = get_llm()

    # Step 1: Classify issue
    classification_chain = CLASSIFICATION_PROMPT | llm

    classification_response = classification_chain.invoke(
        {
            "customer_message": customer_message,
            "customer_name": customer_name,
            "customer_type": customer_type,
            "order_id": order_id
        }
    )

    classification = extract_json(classification_response.content)

    issue_category = classification.get("issue_category", "Other")

    # Step 2: Get order details from CSV
    order_details = get_order_details(order_id)

    if order_details is None:
        order_details_text = json.dumps(
            {
                "found": False,
                "message": "No order found. Ask the customer to verify the order ID."
            },
            indent=2
        )
    else:
        order_details_text = json.dumps(
            order_details,
            indent=2,
            default=str
        )

    # Step 3: Retrieve relevant policy from PDF
    policy_text = get_relevant_policy(
        issue_category=issue_category,
        customer_message=customer_message
    )

    # Step 4: Generate final support response
    final_chain = FINAL_RESPONSE_PROMPT | llm

    final_response = final_chain.invoke(
        {
            "customer_name": customer_name,
            "customer_type": customer_type,
            "order_id": order_id,
            "customer_message": customer_message,
            "classification": json.dumps(classification, indent=2),
            "order_details": order_details_text,
            "policy_text": policy_text
        }
    )

    return {
        "classification": classification,
        "order_details": order_details_text,
        "policy_text": policy_text,
        "final_answer": final_response.content
    }