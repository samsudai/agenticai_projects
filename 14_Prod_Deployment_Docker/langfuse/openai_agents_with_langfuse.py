from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Optional: make sure tracing is not disabled
os.environ.pop("OTEL_SDK_DISABLED", None)
os.environ["LANGFUSE_TRACING_ENABLED"] = "true"

from openai import OpenAI
from langfuse import get_client

# -------------------------
# Clients
# -------------------------
client = OpenAI()
langfuse = get_client()

# -------------------------
# Input
# -------------------------
question = "Explain AI in simple terms"
system_prompt = "Explain concepts in simple, beginner-friendly language."

# -------------------------
# Langfuse observation
# -------------------------
with langfuse.start_as_current_observation(
    as_type="generation",
    name="chat-completion",
    model="gpt-4o-mini",
    input={
        "system": system_prompt,
        "user": question
    }
) as generation:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    print(answer)

    generation.update(
        output={"answer": answer},
        usage={
            "input": response.usage.prompt_tokens if response.usage else 0,
            "output": response.usage.completion_tokens if response.usage else 0,
            "total": response.usage.total_tokens if response.usage else 0
        }
    )

# -------------------------
# Flush for short scripts
# -------------------------
langfuse.flush()