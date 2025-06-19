from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"  # works just like OpenAI
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pdfv-ai-chatbot-hosttest.vercel.app/"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageInput(BaseModel):
    user_input: str

@app.post("/chat")
def get_ai_response(data: MessageInput):
    messages = [
        {
            "role": "system",
            "content": "You are a recruiter for a software company. Your task is to give your thoughts and opinions on how the user carries themselves and make a decision based on their interaction with you."
        },
        {
            "role": "user",
            "content": data.user_input
        }
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        assistant_response = response.choices[0].message.content.strip()
        return {"response": assistant_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
