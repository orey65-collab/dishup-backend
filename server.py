import os
import base64
import json
import re
import httpx
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

app = FastAPI(title="DishUp API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

API_KEY = os.environ.get("EMERGENT_LLM_KEY")

# Proviamo il modello che Google ti ha mostrato nel Quick Start
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={API_KEY}"

class ImageAnalysisRequest(BaseModel):
    image_base64: str

class RecipeRequest(BaseModel):
    ingredients: List[str]
    course_type: str = "primo"
    language: str = "it"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        # Payload ottimizzato per Gemini 2.0
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica gli ingredienti alimentari in questa foto. Rispondi SOLO con un JSON: {\"ingredients\": [\"nome\", \"nome\"]}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}
                ]
            }]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()

        if "candidates" in result:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(json_match.group()) if json_match else {"ingredients": []}
        
        print(f"Errore Google API: {result}")
        raise Exception(result.get("error", {}).get("message", "Modello non trovato o Quota esaurita"))

    except Exception as e:
        print(f"ERRORE ANALISI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    try:
        prompt = f"Crea 3 ricette in {request.language} per {request.course_type} con: {', '.join(request.ingredients)}. Rispondi SOLO JSON: {{ \"recipes\": [ {{ \"title\": \"\", \"prep_time\": 0, \"difficulty\": \"\", \"ingredients\": [], \"steps\": [] }} ] }}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()

        if "candidates" in result:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(json_match.group())
        return {"recipes": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
