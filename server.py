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

# Configurazione Log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

load_dotenv()

app = FastAPI(title="DishUp API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURAZIONE API KEY E URL ---
# Assicurati che su Render la variabile si chiami EMERGENT_LLM_KEY
API_KEY = os.environ.get("EMERGENT_LLM_KEY")

# CORREZIONE URL: Rimosso il parametro key dall'URL string per iniettarlo come query param pulito
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

user_usage = {}

class ImageAnalysisRequest(BaseModel):
    image_base64: str

class DietaryFilters(BaseModel):
    vegetarian: bool = False
    vegan: bool = False
    gluten_free: bool = False

class RecipeRequest(BaseModel):
    user_id: str = "guest"
    ingredients: List[str]
    course_type: str = "primo"
    language: str = "it"
    gym_goal: str = "none"
    is_premium: bool = False
    dietary: Optional[DietaryFilters] = None # Aggiunto per matchare il frontend

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "DishUp is cooking!"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica gli ingredienti alimentari in questa foto. Rispondi solo in JSON: {\"ingredients\": [\"item1\", \"item2\"]}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64_data}}
                ]
            }]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Passiamo la chiave come parametro query per evitare errori di parsing dell'URL
            response = await client.post(BASE_URL, params={"key": API_KEY}, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Gemini Error {response.status_code}: {response.text}")
                raise HTTPException(status_code=502, detail="L'AI non risponde correttamente")
            
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(json_match.group()) if json_match else {"ingredients": []}

    except Exception as e:
        logger.error(f"ERRORE ANALISI: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore nel processare l'immagine")

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    today = str(date.today())
    user_key = f"{request.user_id}_{today}"
    
    # Logica Limite (1 ricetta al giorno per Free)
    if not request.is_premium:
        usage = user_usage.get(user_key, 0)
        if usage >= 1: # Limite a 1 per testare
            raise HTTPException(status_code=403, detail="Limite raggiunto")
        user_usage[user_key] = usage + 1

    try:
        # Costruzione istruzioni dieta
        diet_instr = ""
        if request.dietary:
            if request.dietary.vegan: diet_instr += " La ricetta deve essere VEGANA."
            elif request.dietary.vegetarian: diet_instr += " La ricetta deve essere VEGETARIANA."
            if request.dietary.gluten_free: diet_instr += " La ricetta deve essere SENZA GLUTINE."

        prompt = f"""
        Crea 3 ricette per {request.course_type} con: {', '.join(request.ingredients)}.{diet_instr}
        Lingua: {request.language}. Obiettivo: {request.gym_goal}.
        Rispondi solo in JSON con questo schema:
        {{ "recipes": [ {{ "title": "...", "calories": 0, "ingredients": [], "steps": [] }} ] }}
        """

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(BASE_URL, params={"key": API_KEY}, json=payload)
            
            # Se Gemini fallisce, ora restituiamo un errore vero, non un 200 finto
            if response.status_code != 200:
                logger.error(f"Gemini API Failure: {response.text}")
                raise HTTPException(status_code=502, detail="Errore generazione AI")

            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                if not request.is_premium:
                    data["remaining_limit"] = 0
                return data
            
            raise ValueError("Incompatibilità formato JSON AI")

    except Exception as e:
        logger.error(f"ERRORE GENERAZIONE: {str(e)}")
        # Importante: ora il frontend riceverà un errore 500 se qualcosa va storto
        raise HTTPException(status_code=500, detail=str(e))
