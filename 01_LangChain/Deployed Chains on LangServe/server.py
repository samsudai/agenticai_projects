#!/usr/bin/env python

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langserve import add_routes
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Create a simple joke generator prompt
system_template = "You are a helpful assistant that generates jokes about {topic}."
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', 'Tell me a short, clean joke about {topic}.')
])

# 2. Create model (using Google's Gemini 1.5 Flash)
model = ChatOpenAI(
                model=("gpt-4o-mini")
            )

# 3. Create parser to handle the output
parser = StrOutputParser()

# 4. Create the chain by connecting components
chain = prompt_template | model | parser

# 5. Create the FastAPI application
app = FastAPI(
    title="Joke Generator API",
    version="1.0",
    description="A simple demo API using LangChain and LangServe to generate jokes about a given topic",
)

# 6. Add routes for the chain
add_routes(
    app,
    chain,
    path="/joke-generator",
)

# 7. Add a simple homepage
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Joke Generator API",
        "endpoints": {
            "joke_generator": "/joke-generator",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting the Joke Generator API server...")
    print("Visit http://localhost:8000/docs to interact with the API")
    uvicorn.run(app, host="localhost", port=8000)