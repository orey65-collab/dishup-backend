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
from google import genai

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

client = genai.Client(api_key=os.environ.get("EMERGENT_LLM_KEY"))
# Usiamo 1.5 Flash perché ha quote gratuite garantite
MODEL_ID = "gemini-1.5-flash" 

class ImageAnalysisRequest(BaseModel):
    image_base64: str

class RecipeRequest(BaseModel):
    ingredients: List[str]
    course_type: str = "primo"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                "Identifica gli ingredienti in questa foto. Rispondi SOLO in JSON: {\"ingredients\": [\"nome\"]}",
                genai.types.Part.from_bytes(
                    data=base64.b64decode(base64_data),
                    mime_type="image/jpeg"
                )
            ]
        )

        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"ingredients": []}

    except Exception as e:
        print(f"Errore Analisi: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    try:
        prompt = f"Crea 3 ricette per {request.course_type} con: {', '.join(request.ingredients)}. Rispondi SOLO JSON: {{ \"recipes\": [ {{ \"title\": \"\", \"prep_time\": 0, \"difficulty\": \"\", \"ingredients\": [], \"steps\": [] }} ] }}"
        
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        return json.loads(json_match.group()) if json_match else {"recipes": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
