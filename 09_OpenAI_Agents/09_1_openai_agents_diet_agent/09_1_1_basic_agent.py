from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, trace

load_dotenv(override=True)

nutrition_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful assistant giving out nutrition advice.
    You give concise answers.
    """,
)

async def main():
    # See platform.openai.com/logs?api=traces to see results after execution
    with trace("Simple Nutrition Agent"):
        result = await Runner.run(
            nutrition_agent,
            "How healthy are bananas?"
        )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
