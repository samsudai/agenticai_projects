from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, function_tool, trace, ModelSettings

load_dotenv(override=True)

@function_tool
async def get_food_calories(food_item: str) -> str:
    """
    Get calorie information for common foods to help with nutrition tracking.
    """
    calorie_data = {
        "apple": "80 calories per medium apple (182g)",
        "banana": "105 calories per medium banana (118g)",
        "broccoli": "25 calories per 1 cup chopped (91g)",
        "almonds": "164 calories per 1oz (28g) or about 23 nuts",
    }

    food_key = food_item.lower()
    if food_key in calorie_data:
        return f"{food_item.title()}: {calorie_data[food_key]}"
    else:
        return f"I don't have calorie data for {food_item}."



calorie_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful nutrition assistant giving out calorie information.
    You give concise answers.
    """,
    tools=[get_food_calories],
    model_settings=ModelSettings(tool_choice="get_food_calories"),
)

async def main():
    with trace("Nutrition Assistant with tools"):
        result = await Runner.run(
            calorie_agent,
            "How many calories are in total in a banana and an apple?"
        )
        print(result.final_output)

asyncio.run(main())
