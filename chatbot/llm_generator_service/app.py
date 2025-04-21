# chatbot/llm_generator_service/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OPENAI_API_KEY environment variable")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Guardrail configuration
GUARDRAIL_TEXT = os.getenv(
    "GUARDRAIL_TEXT",
    "You are an AI assistant for DeepMed, a no-code AI platform for medical professionals. Your purpose is to help users understand and use the DeepMed platform — including how to upload data, configure models, interpret results, and navigate the interface. You are also allowed to explain relevant AI concepts **as they apply to DeepMed**, such as why small datasets might lead to poor accuracy, or what a classification model does. However, you must not act like a general-purpose AI tutor. Only explain AI in the context of how DeepMed uses it. You must never provide medical advice, diagnoses, or treatment suggestions under any circumstances. If a user asks for such help, respond with: \"I’m not qualified to answer that. Please consult a healthcare professional.\" You must also not answer any questions unrelated to DeepMed. Even if the user says it's relevant or claims it will help them use the site, do not respond unless you know it is within the scope of DeepMed. You know exactly what DeepMed does. If something is not part of the platform, clearly say: That is not part of DeepMed’s functionality. Always stay focused on helping users safely and effectively use DeepMed and its built-in AI features."
)


client = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized")

class Message(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.2

class LLMResponse(BaseModel):
    reply: str

@app.post("/llm/generate", response_model=LLMResponse)
async def generate(llm_req: LLMRequest):
    try:
        # Apply guardrail to the system message or add one if it doesn't exist
        has_system_message = False
        for msg in llm_req.messages:
            if msg.role == "system":
                msg.content = f"{msg.content}\n\n{GUARDRAIL_TEXT}"
                has_system_message = True
                break
        
        # If no system message exists, prepend one with the guardrail
        if not has_system_message:
            llm_req.messages.insert(0, Message(role="system", content=GUARDRAIL_TEXT))
        
        logger.info(f"Generating response with {len(llm_req.messages)} messages (with guardrail)")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m.role, "content": m.content} for m in llm_req.messages],
            max_tokens=llm_req.max_tokens,
            temperature=llm_req.temperature
        )
        reply = resp.choices[0].message.content
        logger.info(f"Successfully generated reply: {reply[:50]}...")
        return LLMResponse(reply=reply)
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm_generator"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LLM generator service on port 5203")
    uvicorn.run("app:app", host="0.0.0.0", port=5203, reload=True)
