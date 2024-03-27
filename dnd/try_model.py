from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import httpx

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/v1/chat/completions")
async def get_completion(prompt: Prompt):
    # Define the URL of the backend API
    url = "http://localhost:8000/v1/completions"

    # Define the headers for the request
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    # Define the data for the request
    data = {
        "prompt": prompt.prompt,
        "stop": ["\n", "###"],
        "max_tokens": 1000,
    }

    # Send the request to the backend API and get the response
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)

    # Return the response data
    return response.json()


async def main():
    print(await get_completion(Prompt(prompt="Your name is alice. Let's roleplay")))

asyncio.run(main())