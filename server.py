import os
import base64
import json
import re
import httpx
import logging
from datetime import date
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Configurazione Log per vedere gli errori su Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="DishUp API")

# --- CONFIGURAZIONE CORS OTTIMIZZATA ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("EMERGENT_LLM_KEY")
# Usiamo l'ultima versione stabile del modello per evitare deprecazioni
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# DATABASE TEMPORANEO
user_usage = {}

class ImageAnalysisRequest(BaseModel):
    image_base64: str

class RecipeRequest(BaseModel):
    user_id: str = "guest"
    ingredients: List[str]
    course_type: str = "primo"
    language: str = "it"
    gym_goal: str = "none"
    is_premium: bool = False

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "DishUp is cooking!"}

# --- MIDDLEWARE PER LOGGING (Aiuta a capire se lo smartphone invia dati) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        # Pulizia stringa Base64 (rimuove header data:image/jpeg;base64,)
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        # Log della dimensione per debug
        logger.info(f"Processing image: {len(base64_data)} bytes")

        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica SOLO gli ingredienti alimentari presenti in questa foto. Rispondi esclusivamente in formato JSON: {\"ingredients\": [\"mela\", \"farina\"]}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}
                ]
            }]
        }

        # Timeout esteso a 60 secondi perché Gemini con immagini può essere lento
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Gemini API Error: {response.text}")
                raise HTTPException(status_code=502, detail="Errore dal servizio di intelligenza artificiale")
            
            result = response.json()
            
            if "candidates" in result:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Estrazione robusta del JSON
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            return {"ingredients": []}

    except Exception as e:
        logger.error(f"ERRORE ANALISI: {str(e)}")
        raise HTTPException(status_code=500, detail="Il server non è riuscito a processare l'immagine")

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    today = str(date.today())
    user_key = f"{request.user_id}_{today}"
    
    # Logica Limite Freemium
    if not request.is_premium:
        usage = user_usage.get(user_key, 0)
        if usage >= 2:
            raise HTTPException(
                status_code=403, 
                detail="Limite giornaliero raggiunto. Passa a Premium!"
            )
        user_usage[user_key] = usage + 1

    try:
        # Configurazione Prompt in base agli obiettivi
        fitness_instr = ""
        if request.gym_goal == "bulk":
            fitness_instr = "Obiettivo BULK: Ricette ipercaloriche, ricche di proteine e carboidrati. Includi macros precisi."
        elif request.gym_goal == "cut":
            fitness_instr = "Obiettivo CUT: Ricette ipocaloriche, alto volume e alto contenuto proteico. Includi macros precisi."
        elif request.gym_goal == "none" and request.is_premium:
            fitness_instr = "Includi macros dettagliati (proteine, carboidrati, grassi)."
        else:
            fitness_instr = "Fornisci solo le calorie totali. Non mostrare i macros dettagliati."

        prompt = f"""
        Crea 3 ricette creative per {request.course_type} usando questi ingredienti: {', '.join(request.ingredients)}.
        Lingua: {request.language}.
        {fitness_instr}

        Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. 
        Schema:
        {{
          "recipes": [
            {{
              "title": "Titolo",
              "prep_time": 30,
              "difficulty": "facile",
              "servings": 2,
              "calories": 450,
              "macros": {{ "protein": 30, "carbs": 50, "fat": 15 }},
              "special_reason": "✨ Perché è perfetta",
              "ingredients": [{{ "name": "...", "quantity": "..." }}],
              "steps": ["..."],
              "wine_pairing": {{ "wine": "...", "description": "..." }},
              "bon_appetit": "Buon appetito!"
            }}
          ]
        }}
        """

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()
            
            if "candidates" in result:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    recipe_data = json.loads(json_match.group())
                    if not request.is_premium:
                        recipe_data["remaining_limit"] = 2 - user_usage[user_key]
                    return recipe_data
            
            return {"recipes": [], "error": "AI failed to generate recipes"}

    except Exception as e:
        logger.error(f"ERRORE RICETTA: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore nella creazione della ricetta")

if __name__ == "__main__":
    import uvicorn
    # Render assegna la porta dinamicamente
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
