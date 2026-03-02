import os
import base64
import json
import re
import httpx
from datetime import date
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DishUp API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("EMERGENT_LLM_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={API_KEY}"

# --- DATABASE TEMPORANEO IN MEMORIA ---
# In produzione andrebbe sostituito con Redis o Supabase
user_usage = {}

class ImageAnalysisRequest(BaseModel):
    image_base64: str

class RecipeRequest(BaseModel):
    user_id: str = "guest"  # Identificativo unico utente
    ingredients: List[str]
    course_type: str = "primo"
    language: str = "it"
    gym_goal: str = "none"  # "bulk", "cut", o "none"
    is_premium: bool = False

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
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
    # --- LOGICA LIMITE FREEMIUM ---
    today = str(date.today())
    if not request.is_premium:
        user_key = f"{request.user_id}_{today}"
        usage = user_usage.get(user_key, 0)
        if usage >= 2:
            raise HTTPException(
                status_code=403, 
                detail="Limite giornaliero raggiunto. Passa a Premium per ricette illimitate!"
            )
        user_usage[user_key] = usage + 1

    try:
        # --- CONFIGURAZIONE PROMPT GYMRAT ---
        fitness_instr = ""
        if request.is_premium:
            if request.gym_goal == "bulk":
                fitness_instr = "Obiettivo BULK: Ricette ipercaloriche, ricche di proteine e carboidrati. Includi macros dettagliati."
            elif request.gym_goal == "cut":
                fitness_instr = "Obiettivo CUT: Ricette ipocaloriche, alto volume/fibre e alto contenuto proteico. Includi macros dettagliati."
            else:
                fitness_instr = "Includi calorie e macronutrienti (proteine, carbi, grassi) per ogni ricetta."
        else:
            fitness_instr = "Fornisci solo le calorie totali approssimative. Non includere i macros dettagliati."

        prompt = f"""
        Crea 3 ricette per {request.course_type} usando questi ingredienti: {', '.join(request.ingredients)}.
        Lingua: {request.language}.
        {fitness_instr}

        Rispondi RIGOROSAMENTE con questo schema JSON:
        {{
          "recipes": [
            {{
              "title": "Titolo",
              "prep_time": 30,
              "difficulty": "facile",
              "servings": 1,
              "calories": 0,
              "macros": {{ "protein": 0, "carbs": 0, "fat": 0 }},
              "special_reason": "✨ Perché è perfetta per te",
              "ingredients": [{{ "name": "...", "quantity": "..." }}],
              "steps": ["..."],
              "wine_pairing": {{ "wine": "...", "description": "..." }},
              "bon_appetit": "🍽️"
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
                recipe_data = json.loads(json_match.group())
                
                # Aggiungiamo info sul limite residuo per l'utente free
                if not request.is_premium:
                    recipe_data["remaining_limit"] = 2 - user_usage[user_key]
                
                return recipe_data
            
            return {"recipes": []}

    except Exception as e:
        print(f"ERRORE RICETTA: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore nella creazione della ricetta")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
