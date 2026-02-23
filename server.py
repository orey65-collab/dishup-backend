import os
import base64
import json
import re
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DishUp API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("EMERGENT_LLM_KEY")
# Continuiamo con il modello che ha funzionato
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={API_KEY}"

class ImageAnalysisRequest(BaseModel):
    image_base64: str

class RecipeRequest(BaseModel):
    ingredients: List[str]
    course_type: str = "primo"

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica gli ingredienti alimentari. Rispondi SOLO JSON: {\"ingredients\": [\"nome\"]}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}
                ]
            }]
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(re.search(r'\{.*\}', text, re.DOTALL).group())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    try:
        # Prompt istruito per rispondere esattamente con i tuoi campi
        prompt = f"""
        Crea 3 ricette per {request.course_type} usando: {', '.join(request.ingredients)}.
        Rispondi in ITALIANO.
        Per ogni ricetta segui RIGOROSAMENTE questo schema JSON:
        {{
          "recipes": [
            {{
              "title": "Nome della ricetta",
              "prep_time": "45 min",
              "difficulty": "Media",
              "special_reason": "Spiega perché questa ricetta è speciale (es: unione di sapori, leggerezza, originalità)",
              "ingredients_list": ["lista dettagliata con dosi"],
              "procedure": ["passaggio 1", "passaggio 2", "passaggio 3"],
              "sommelier_advice": "Nome del vino e perché si abbina perfettamente a questo piatto"
            }}
          ]
        }}
        Rispondi solo con il JSON.
        """
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(re.search(r'\{.*\}', text, re.DOTALL).group())
    except Exception as e:
        print(f"Errore: {e}")
        raise HTTPException(status_code=500, detail="Errore nella generazione della ricetta")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
