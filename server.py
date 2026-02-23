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

# Configurazione CORS per permettere al frontend di comunicare col server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("EMERGENT_LLM_KEY")
# Modello Gemini 3 Flash Preview - L'unico sbloccato per il tuo account nel 2026
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
        # Pulizia della stringa base64
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica SOLO gli ingredienti alimentari presenti. Rispondi esclusivamente in formato JSON: {\"ingredients\": [\"mela\", \"farina\"]}"},
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
                return json.loads(json_match.group())
            
            raise Exception("Risposta Google non valida")

    except Exception as e:
        print(f"ERRORE ANALISI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    try:
        # PROMPT OTTIMIZZATO PER IL TUO RecipeScreen.js
        prompt = f"""
        Crea 3 ricette per {request.course_type} usando questi ingredienti: {', '.join(request.ingredients)}.
        La lingua deve essere {request.language}.
        Rispondi RIGOROSAMENTE con un oggetto JSON che segue questa struttura:
        {{
          "recipes": [
            {{
              "title": "Titolo Accattivante",
              "prep_time": 45,
              "difficulty": "media",
              "servings": 2,
              "calories": 400,
              "special_reason": "Spiega perché è speciale (✨)",
              "ingredients": [
                {{ "name": "nome ingrediente", "quantity": "dose es. 200g" }}
              ],
              "steps": ["Passaggio 1", "Passaggio 2"],
              "wine_pairing": {{
                "wine": "Nome del Vino (🍷)",
                "description": "Perché si abbina bene"
              }},
              "bon_appetit": "Augurio finale (🍽️)"
            }}
          ]
        }}
        Non aggiungere testo prima o dopo il JSON.
        """

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()
            
            if "candidates" in result:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                return json.loads(json_match.group())
            
            return {"recipes": []}

    except Exception as e:
        print(f"ERRORE RICETTA: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore nella creazione della ricetta")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
