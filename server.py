
import os
import base64
import json
import re
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="DishUp API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# --- CONFIGURAZIONE GOOGLE GEMINI ---
api_key = os.environ.get("EMERGENT_LLM_KEY")

# Forza l'uso della versione stabile V1 invece della beta
os.environ["GOOGLE_GENERATIVE_AI_NETWORK_ENDPOINT"] = "generativelanguage.googleapis.com:443"

genai.configure(api_key=api_key, transport='rest')

# Usiamo il nome del modello senza suffissi 'latest' o 'v1beta'
MODEL_NAME = 'gemini-1.5-flash'

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    language: str = "it"

class RecipeGenerationRequest(BaseModel):
    ingredients: List[str]
    course_type: str = "primo"
    quick_recipe: bool = False
    gourmet: bool = False
    language: str = "it"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        # 1. Pulizia e decodifica base64
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        image_bytes = base64.b64decode(base64_data)
        
        # 2. Ottimizzazione immagine per Render
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB': 
                img = img.convert('RGB')
            img.thumbnail((800, 800)) 
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_content = buffered.getvalue()

        # 3. Chiamata a Gemini
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        prompt = "Identifica gli ingredienti alimentari in questa foto. Rispondi SOLO con un JSON: {\"ingredients\": [\"nome\", \"nome\"]}"
        
        response = model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': img_content}
        ])
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"ingredients": []}

    except Exception as e:
        print(f"ERRORE ANALISI: {str(e)}")
        # Se l'errore persiste, stampiamo anche il tipo di errore nei log
        raise HTTPException(status_code=500, detail=f"Errore scansione: {str(e)}")

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeGenerationRequest):
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        
        prompt = f"""Crea 3 ricette {request.language} per {request.course_type} usando: {', '.join(request.ingredients)}. 
        Rispondi SOLO in JSON con questa struttura:
        {{ "recipes": [ {{ "title": "", "prep_time": 20, "difficulty": "facile", "ingredients": [], "steps": [] }} ] }}"""

        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        return {"recipes": []}

    except Exception as e:
        print(f"ERRORE RICETTA: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore generazione ricetta")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
