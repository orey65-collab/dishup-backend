import os
import uuid
import base64
import json
import re
import asyncio
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from PIL import Image
import pillow_heif
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

# Configurazione Google Gemini
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")
genai.configure(api_key=EMERGENT_LLM_KEY)

# --- COSTANTI E TRADUZIONI (MANTENUTE DAL TUO ORIGINALE) ---
ITALIAN_INGREDIENTS = [
    "pomodori", "basilico", "mozzarella", "parmigiano", "pecorino", "olio d'oliva",
    "aglio", "cipolla", "carote", "sedano", "patate", "zucchine", "melanzane",
    "peperoni", "spinaci", "rucola", "lattuga", "funghi", "olive", "capperi",
    "pasta", "spaghetti", "penne", "fusilli", "rigatoni", "lasagne", "gnocchi",
    "riso", "risotto", "farina", "lievito", "uova", "burro", "latte", "panna",
    "prosciutto", "pancetta", "guanciale", "salsiccia", "pollo", "manzo",
    "maiale", "vitello", "agnello", "pesce", "tonno", "salmone", "gamberi",
    "vongole", "cozze", "calamari", "acciughe", "limone", "arancia", "mela",
    "fragole", "lamponi", "mirtilli", "banana", "pesca", "albicocca",
    "zucchero", "miele", "cioccolato", "vaniglia", "cannella", "noce moscata",
    "pepe", "sale", "origano", "rosmarino", "timo", "prezzemolo", "salvia",
    "vino bianco", "vino rosso", "aceto balsamico", "brodo", "passata di pomodoro"
]

TRANSLATIONS = {
    "it": {
        "scan_ingredients": "Scansiona Ingredienti", "add_ingredients": "Aggiungi ingredienti...",
        "current_pantry": "Dispensa Attuale", "appetizer": "Antipasto", "first_course": "Primo",
        "second_course": "Secondo", "dessert": "Dessert", "quick_recipes": "Ricette Veloci (<20 min)",
        "gourmet_recipes": "Sfidanti/Gourmet", "generate_recipe": "Genera Ricetta Sfiziosa",
        "prep_time": "Tempo di preparazione", "difficulty": "Difficoltà", "why_special": "Perché questa ricetta è speciale",
        "ingredients": "Ingredienti", "instructions": "Istruzioni", "next": "Avanti", "previous": "Indietro",
        "finish": "Fine", "home": "Home", "favorites": "Preferiti", "profile": "Profilo",
        "no_ingredients": "Nessun ingrediente nella dispensa", "scanning": "Analizzando l'immagine...",
        "generating": "Sto creando la tua ricetta...", "easy": "Facile", "medium": "Media", "hard": "Difficile",
        "minutes": "minuti", "servings": "porzioni", "language": "Lingua", "take_photo": "Scatta Foto",
        "choose_gallery": "Scegli dalla Galleria", "cancel": "Annulla", "error_camera": "Errore nell'analisi dell'immagine",
        "add_at_least_one": "Aggiungi almeno un ingrediente", "pantry_cleared": "Dispensa svuotata", "clear_pantry": "Svuota dispensa"
    }
}

# --- SCHEMI DATI ---
class ImageAnalysisRequest(BaseModel):
    image_base64: str
    language: str = "it"

class RecipeGenerationRequest(BaseModel):
    ingredients: List[str]
    course_type: str = "primo"
    quick_recipe: bool = False
    gourmet: bool = False
    vegan: bool = False
    gluten_free: bool = False
    vegetarian: bool = False
    language: str = "it"

class IngredientSearchRequest(BaseModel):
    query: str
    language: str = "it"

# --- ENDPOINTS ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/translations/{language}")
async def get_translations(language: str):
    return {"translations": TRANSLATIONS.get(language, TRANSLATIONS["it"])}

@app.post("/api/ingredients/search")
async def search_ingredients(request: IngredientSearchRequest):
    query = request.query.lower()
    if len(query) < 2: return {"suggestions": []}
    suggestions = [ing for ing in ITALIAN_INGREDIENTS if query in ing.lower()][:8]
    return {"suggestions": suggestions}

@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    if not EMERGENT_LLM_KEY: raise HTTPException(status_code=500, detail="API key missing")
    
    try:
        # Pulizia stringa base64
        base64_data = re.sub(r'^data:image/.+;base64,', '', request.image_base64)
        image_bytes = base64.b64decode(base64_data)
        
        # Gestione formati (HEIC/iPhone)
        pillow_heif.register_heif_opener()
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB': img = img.convert('RGB')
        
        # Ridimensionamento per velocità
        img.thumbnail((1024, 1024))
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        img_encoded = base64.b64encode(buffered.getvalue()).decode()

        # Chiamata a Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        lang_text = "in italiano" if request.language == "it" else "in English"
        
        prompt = f"Sei uno chef. Identifica gli ingredienti alimentari in questa foto. Rispondi SOLO con un JSON: {{\"ingredients\": [\"nome\", \"nome\"]}} {lang_text}."
        
        response = model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': img_encoded}
        ])
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"ingredients": []}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeGenerationRequest):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Costruzione prompt basata sulle tue preferenze
        constraints = []
        if request.quick_recipe: constraints.append("veloce (<20 min)")
        if request.gourmet: constraints.append("gourmet ed elaborata")
        
        prompt = f"""Crea 3 ricette {request.language} per {request.course_type} usando: {', '.join(request.ingredients)}. 
        {f'Vincoli: {", ".join(constraints)}' if constraints else ''}
        Rispondi SOLO in JSON con questa struttura:
        {{ "recipes": [ {{ "title": "", "prep_time": 20, "difficulty": "facile", "servings": 4, "description": "", "ingredients": [{{ "name": "", "quantity": "" }}], "steps": [], "bon_appetit": "", "wine_pairing": {{ "wine": "", "description": "" }} }} ] }}"""

        response = model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group())
        raise Exception("Invalid AI response")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Errore generazione ricetta")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
