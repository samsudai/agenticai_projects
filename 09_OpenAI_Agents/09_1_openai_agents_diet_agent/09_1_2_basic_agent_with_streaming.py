from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, trace
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv(override=True)

nutrition_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful assistant giving out nutrition advice.
    You give concise answers.
    """,
)

async def main():
    # Stream the response
    response_stream = Runner.run_streamed(
        nutrition_agent,
        "How healthy are bananas?"
    )

    # As the response comes in, print it
    async for event in response_stream.stream_events():
        # Only print actual text of the response
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            # Print the delta, no newlines, immediately
            print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
