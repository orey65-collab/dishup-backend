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
from google import genai # Nuova libreria

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

# Inizializzazione Client con la nuova chiave
client = genai.Client(api_key=os.environ.get("EMERGENT_LLM_KEY"))
# Usiamo l'ultimo modello disponibile
MODEL_ID = "gemini-2.0-flash" 

class ImageAnalysisRequest(BaseModel):
    image_base64: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    try:
        # Pulizia base64
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        
        # Analisi con il nuovo metodo SDK
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

        # Estrazione testo e pulizia JSON
        text_response = response.text
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        return {"ingredients": []}

    except Exception as e:
        print(f"Errore DishUp: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Nota: Aggiungi qui l'endpoint per le ricette usando lo stesso 'client.models.generate_content'

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
