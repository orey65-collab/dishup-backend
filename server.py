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

# Configurazione CORS - Fondamentale per far parlare Netlify e Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Recupero chiave API
API_KEY = os.environ.get("EMERGENT_LLM_KEY")

# URL FORZATO alla versione v1 stabile (questo risolve il 404 v1beta)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-001:generateContent?key={API_KEY}"
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
        # 1. Pulizia stringa base64
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        # 2. Ottimizzazione immagine per non saturare i 512MB di Render
        image_bytes = base64.b64decode(base64_data)
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            img.thumbnail((800, 800))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # 3. Payload per Google (formato v1 stabile)
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica gli ingredienti alimentari. Rispondi SOLO JSON: {\"ingredients\": [\"nome\"]}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }]
        }

        # 4. Chiamata diretta con httpx (bypassando le librerie Google)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()

        if "candidates" in result:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(json_match.group()) if json_match else {"ingredients": []}
        
        print(f"Errore dettagliato Google: {result}")
        raise Exception("Risposta Google non valida o Quota esaurita")

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
            return json.loads(json_match.group()) if json_match else {"recipes": []}
        
        return {"recipes": []}
    except Exception as e:
        print(f"ERRORE RICETTA: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore generazione ricetta")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
