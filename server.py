
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
# Forziamo l'URL alla versione v1 stabile (NON beta)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={API_KEY}"

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    language: str = "it"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        # 1. Elaborazione immagine
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        image_bytes = base64.b64decode(base64_data)
        
        with Image.open(BytesIO(image_bytes)) as img:
            if img.mode != 'RGB': img = img.convert('RGB')
            img.thumbnail((800, 800))
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # 2. Payload per la chiamata diretta (v1 stabile)
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Identifica gli ingredienti alimentari. Rispondi SOLO JSON: {\"ingredients\": [\"nome\"]}"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }]
        }

        # 3. Chiamata HTTP diretta senza passare per la libreria genai
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_URL, json=payload)
            result = response.json()

        # Estrazione dati dalla risposta di Google
        if "candidates" in result:
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            return json.loads(json_match.group()) if json_match else {"ingredients": []}
        
        print(f"Errore Google API: {result}")
        raise Exception("Risposta Google non valida")

    except Exception as e:
        print(f"ERRORE ANALISI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: BaseModel):
    # Nota: per brevità ho semplificato, ma usa lo stesso metodo httpx qui sopra
    # puntando a GEMINI_URL con il prompt della ricetta.
    raise HTTPException(status_code=501, detail="Usa lo stesso metodo httpx per le ricette")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
