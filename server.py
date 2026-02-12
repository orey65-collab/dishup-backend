
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Configurazione Google Gemini con la chiave dal pannello Render
genai.configure(api_key=os.environ.get("EMERGENT_LLM_KEY"))

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
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        image_bytes = base64.b64decode(base64_data)
        
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB': 
                img = img.convert('RGB')
            img.thumbnail((800, 800)) 
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_encoded = base64.b64encode(buffered.getvalue()).decode()

        # CORREZIONE 404: Usiamo il nome completo del modello
        model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
        
        prompt = "Identifica gli ingredienti alimentari. Rispondi SOLO JSON: {\"ingredients\": [\"nome\"]}"
        
        response = model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': img_encoded}
        ])
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"ingredients": []}

    except Exception as e:
        print(f"Errore Analisi: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeGenerationRequest):
    try:
        # CORREZIONE 404: Usiamo il nome completo del modello
        model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')
        
        prompt = f"Crea 3 ricette {request.language} per {request.course_type} usando: {', '.join(request.ingredients)}. Rispondi SOLO JSON: {{ \"recipes\": [ {{ \"title\": \"\", \"prep_time\": 20, \"difficulty\": \"facile\", \"ingredients\": [], \"steps\": [] }} ] }}"
        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"recipes": []}
    except Exception as e:
        print(f"Errore Ricetta: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore generazione")
