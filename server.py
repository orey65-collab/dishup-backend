import os
import uuid
import base64
import json
import re
import asyncio
import imghdr
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from PIL import Image
import pillow_heif

load_dotenv()

app = FastAPI(title="DishUp API")

# Increase max request body size to 20MB for image uploads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Note: Uvicorn handles max body size via --limit-request-line flag
# Default is 16KB for headers, but body size is unlimited by default


# Health check endpoint for Kubernetes - MUST be at root level, no prefix
@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes"""
    return {"status": "ok"}


# CORS Configuration
cors_origins = os.environ.get("CORS_ORIGINS", "*")

# Environment variables
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY")

# Common Italian ingredients for autocomplete
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

# Translations
TRANSLATIONS = {
    "it": {
        "scan_ingredients": "Scansiona Ingredienti",
        "add_ingredients": "Aggiungi ingredienti...",
        "current_pantry": "Dispensa Attuale",
        "appetizer": "Antipasto",
        "first_course": "Primo",
        "second_course": "Secondo",
        "dessert": "Dessert",
        "quick_recipes": "Ricette Veloci (<20 min)",
        "gourmet_recipes": "Sfidanti/Gourmet",
        "generate_recipe": "Genera Ricetta Sfiziosa",
        "prep_time": "Tempo di preparazione",
        "difficulty": "Difficoltà",
        "why_special": "Perché questa ricetta è speciale",
        "ingredients": "Ingredienti",
        "instructions": "Istruzioni",
        "next": "Avanti",
        "previous": "Indietro",
        "finish": "Fine",
        "home": "Home",
        "favorites": "Preferiti",
        "profile": "Profilo",
        "no_ingredients": "Nessun ingrediente nella dispensa",
        "scanning": "Analizzando l'immagine...",
        "generating": "Sto creando la tua ricetta...",
        "easy": "Facile",
        "medium": "Media",
        "hard": "Difficile",
        "minutes": "minuti",
        "servings": "porzioni",
        "language": "Lingua",
        "take_photo": "Scatta Foto",
        "choose_gallery": "Scegli dalla Galleria",
        "cancel": "Annulla",
        "error_camera": "Errore nell'analisi dell'immagine",
        "add_at_least_one": "Aggiungi almeno un ingrediente",
        "pantry_cleared": "Dispensa svuotata",
        "clear_pantry": "Svuota dispensa"
    },
    "en": {
        "scan_ingredients": "Scan Ingredients",
        "add_ingredients": "Add ingredients...",
        "current_pantry": "Current Pantry",
        "appetizer": "Appetizer",
        "first_course": "First Course",
        "second_course": "Main Course",
        "dessert": "Dessert",
        "quick_recipes": "Quick Recipes (<20 min)",
        "gourmet_recipes": "Gourmet/Challenging",
        "generate_recipe": "Generate Tasty Recipe",
        "prep_time": "Prep Time",
        "difficulty": "Difficulty",
        "why_special": "Why this recipe is special",
        "ingredients": "Ingredients",
        "instructions": "Instructions",
        "next": "Next",
        "previous": "Previous",
        "finish": "Finish",
        "home": "Home",
        "favorites": "Favorites",
        "profile": "Profile",
        "no_ingredients": "No ingredients in pantry",
        "scanning": "Analyzing image...",
        "generating": "Creating your recipe...",
        "easy": "Easy",
        "medium": "Medium",
        "hard": "Hard",
        "minutes": "minutes",
        "servings": "servings",
        "language": "Language",
        "take_photo": "Take Photo",
        "choose_gallery": "Choose from Gallery",
        "cancel": "Cancel",
        "error_camera": "Error analyzing image",
        "add_at_least_one": "Add at least one ingredient",
        "pantry_cleared": "Pantry cleared",
        "clear_pantry": "Clear pantry"
    }
}


class ImageAnalysisRequest(BaseModel):
    image_base64: str
    language: str = "it"


class RecipeGenerationRequest(BaseModel):
    ingredients: List[str]
    course_type: str = "primo"  # antipasto, primo, secondo, dessert
    quick_recipe: bool = False
    gourmet: bool = False
    vegan: bool = False
    gluten_free: bool = False
    vegetarian: bool = False
    language: str = "it"


class IngredientSearchRequest(BaseModel):
    query: str
    language: str = "it"


@app.get("/api/")
async def root():
    return {"message": "SvuotaFrigo AI API - Benvenuto!"}


@app.get("/api/translations/{language}")
async def get_translations(language: str):
    """Get UI translations for the specified language"""
    if language not in TRANSLATIONS:
        language = "it"
    return {"translations": TRANSLATIONS[language]}


@app.post("/api/ingredients/search")
async def search_ingredients(request: IngredientSearchRequest):
    """Autocomplete search for ingredients"""
    query = request.query.lower()
    if len(query) < 2:
        return {"suggestions": []}
    
    suggestions = [
        ing for ing in ITALIAN_INGREDIENTS 
        if query in ing.lower()
    ][:8]
    
    return {"suggestions": suggestions}


@app.post("/api/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze an image using Gemini Vision to detect food ingredients"""
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    try:
        print(f"[ANALYZE-IMAGE] Starting analysis, language: {request.language}")
        
        # Decode base64 to get image bytes
        try:
            image_bytes = base64.b64decode(request.image_base64)
            image_size_kb = len(image_bytes) / 1024
            print(f"[ANALYZE-IMAGE] Original image size: {image_size_kb:.2f} KB")
            
            # Detect image type
            detected_type = imghdr.what(None, h=image_bytes)
            print(f"[ANALYZE-IMAGE] Detected image type: {detected_type}")
            
            # Convert HEIC to JPEG automatically
            if detected_type == 'heic' or not detected_type:
                # Try to open with pillow-heif
                print(f"[ANALYZE-IMAGE] HEIC detected, converting to JPEG...")
                try:
                    # Register HEIF opener
                    pillow_heif.register_heif_opener()
                    
                    # Open image (works with HEIC)
                    img = Image.open(BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large (max 1920x1080)
                    max_width, max_height = 1920, 1080
                    if img.width > max_width or img.height > max_height:
                        ratio = min(max_width / img.width, max_height / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        print(f"[ANALYZE-IMAGE] Resized from {img.width}x{img.height} to {new_size[0]}x{new_size[1]}")
                    
                    # Convert to JPEG with good quality
                    output = BytesIO()
                    img.save(output, format='JPEG', quality=85, optimize=True)
                    output.seek(0)
                    
                    # Update image bytes and encode to base64
                    converted_bytes = output.read()
                    request.image_base64 = base64.b64encode(converted_bytes).decode('utf-8')
                    
                    converted_size_kb = len(converted_bytes) / 1024
                    print(f"[ANALYZE-IMAGE] HEIC converted to JPEG: {converted_size_kb:.2f} KB")
                    
                except Exception as heic_error:
                    print(f"[ANALYZE-IMAGE] HEIC conversion failed: {heic_error}")
                    raise HTTPException(
                        status_code=400,
                        detail="Impossibile convertire HEIC. Usa JPEG o PNG." if request.language == "it" else "Cannot convert HEIC. Use JPEG or PNG."
                    )
            
            # Also optimize other large images
            elif image_size_kb > 2000:  # > 2MB
                print(f"[ANALYZE-IMAGE] Large image detected, optimizing...")
                try:
                    img = Image.open(BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    max_width, max_height = 1920, 1080
                    if img.width > max_width or img.height > max_height:
                        ratio = min(max_width / img.width, max_height / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        print(f"[ANALYZE-IMAGE] Resized to {new_size[0]}x{new_size[1]}")
                    
                    # Re-encode as JPEG with good quality
                    output = BytesIO()
                    img.save(output, format='JPEG', quality=85, optimize=True)
                    output.seek(0)
                    
                    optimized_bytes = output.read()
                    request.image_base64 = base64.b64encode(optimized_bytes).decode('utf-8')
                    
                    optimized_size_kb = len(optimized_bytes) / 1024
                    print(f"[ANALYZE-IMAGE] Optimized to {optimized_size_kb:.2f} KB")
                    
                except Exception as opt_error:
                    print(f"[ANALYZE-IMAGE] Optimization warning: {opt_error}, using original")
            
        except Exception as e:
            print(f"[ANALYZE-IMAGE] Image processing error: {e}, proceeding with original")
        
        session_id = f"scan_{uuid.uuid4().hex[:8]}"
        
        lang_instruction = "in italiano" if request.language == "it" else "in English"
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_id,
            system_message=f"""Sei un esperto chef italiano che analizza immagini di ingredienti alimentari.
            Identifica tutti gli ingredienti visibili nell'immagine.
            Rispondi SOLO con una lista JSON di ingredienti {lang_instruction}.
            Formato: {{"ingredients": ["ingrediente1", "ingrediente2", ...]}}
            Concentrati solo su ingredienti alimentari (non utensili o contenitori).
            Se non riesci a identificare ingredienti, rispondi con {{"ingredients": []}}"""
        ).with_model("gemini", "gemini-2.5-flash")
        
        print(f"[ANALYZE-IMAGE] Creating ImageContent with processed base64 data...")
        image_content = ImageContent(image_base64=request.image_base64)
        
        print(f"[ANALYZE-IMAGE] Creating UserMessage...")
        user_message = UserMessage(
            text=f"Analizza questa immagine e identifica tutti gli ingredienti alimentari visibili. Rispondi {lang_instruction} con una lista JSON.",
            file_contents=[image_content]
        )
        
        print(f"[ANALYZE-IMAGE] Sending to Gemini (max 25s)...")
        
        # Send with timeout
        try:
            response = await asyncio.wait_for(
                chat.send_message(user_message),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            print(f"[ANALYZE-IMAGE] ERROR: Timeout after 25s")
            raise HTTPException(
                status_code=504, 
                detail="Timeout AI. Riprova con un'immagine più semplice." if request.language == "it" else "AI timeout. Try a simpler image."
            )
        
        print(f"[ANALYZE-IMAGE] Response received: {response[:200]}...")
        
        # Parse response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                ingredients = result.get("ingredients", [])
                print(f"[ANALYZE-IMAGE] Extracted {len(ingredients)} ingredients from JSON")
            except json.JSONDecodeError as je:
                print(f"[ANALYZE-IMAGE] JSON parse failed: {je}")
                ingredients = []
        else:
            print(f"[ANALYZE-IMAGE] No JSON, extracting from text")
            ingredients = [line.strip().strip('-•').strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('{')]
            ingredients = [ing for ing in ingredients if len(ing) > 1 and len(ing) < 50]
        
        # Filter
        filtered_ingredients = [
            ing for ing in ingredients 
            if ing.lower() not in ['ingredients', 'ingredienti', 'lista', 'list']
        ]
        
        print(f"[ANALYZE-IMAGE] Returning {len(filtered_ingredients)} ingredients: {filtered_ingredients[:10]}...")
        return {"ingredients": filtered_ingredients[:15]}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ANALYZE-IMAGE] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_str = str(e).lower()
        if '502' in error_str or 'bad gateway' in error_str or 'badgatewayerror' in error_str:
            raise HTTPException(
                status_code=502, 
                detail="AI non disponibile. Riprova." if request.language == "it" else "AI unavailable. Try again."
            )
        
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")


@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeGenerationRequest):
    """Generate a recipe using Gemini based on available ingredients"""
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="Nessun ingrediente fornito")
    
    try:
        session_id = f"recipe_{uuid.uuid4().hex[:8]}"
        
        # Map course types to Italian
        course_names = {
            "antipasto": "un antipasto",
            "primo": "un primo piatto",
            "secondo": "un secondo piatto",
            "dessert": "un dessert"
        }
        course = course_names.get(request.course_type, "un piatto")
        
        # Build constraints
        constraints = []
        if request.quick_recipe:
            constraints.append("che richieda massimo 20 minuti di preparazione")
        if request.gourmet:
            constraints.append("che sia elaborato e gourmet, degno di uno chef stellato")
        
        # Dietary restrictions
        dietary_restrictions_it = []
        dietary_restrictions_en = []
        if request.vegan:
            dietary_restrictions_it.append("VEGANA (senza carne, pesce, uova, latticini, miele)")
            dietary_restrictions_en.append("VEGAN (no meat, fish, eggs, dairy, honey)")
        if request.vegetarian:
            dietary_restrictions_it.append("VEGETARIANA (senza carne e pesce)")
            dietary_restrictions_en.append("VEGETARIAN (no meat or fish)")
        if request.gluten_free:
            dietary_restrictions_it.append("SENZA GLUTINE (no pasta di grano, pane, farina di frumento)")
            dietary_restrictions_en.append("GLUTEN-FREE (no wheat pasta, bread, wheat flour)")
        
        dietary_text_it = ""
        dietary_text_en = ""
        if dietary_restrictions_it:
            dietary_text_it = f"IMPORTANTE: La ricetta DEVE essere {', '.join(dietary_restrictions_it)}. "
            dietary_text_en = f"IMPORTANT: The recipe MUST be {', '.join(dietary_restrictions_en)}. "
        
        constraint_text = " ".join(constraints) if constraints else ""
        
        lang = request.language
        if lang == "it":
            prompt = f"""Crea ESATTAMENTE 3 ricette diverse per {course} creative e sfiziose {constraint_text} usando principalmente questi ingredienti: {', '.join(request.ingredients)}.

{dietary_text_it}Le 3 ricette devono essere DIVERSE tra loro: una più semplice/classica, una più creativa/moderna, e una più elaborata/gourmet.

Rispondi SOLO in formato JSON valido con questa struttura esatta:
{{
    "recipes": [
        {{
            "title": "Nome creativo della ricetta 1",
            "prep_time": 25,
            "difficulty": "facile|media|difficile",
            "servings": 4,
            "calories": 450,
            "description": "Una breve descrizione del perché questa ricetta è speciale (2-3 frasi)",
            "ingredients": [
                {{"name": "ingrediente", "quantity": "quantità"}}
            ],
            "steps": [
                "Primo passo dettagliato",
                "Secondo passo dettagliato"
            ],
            "bon_appetit": "Una frase elegante e poetica per augurare buon appetito (es: 'Che questo piatto porti gioia alla tua tavola!')",
            "wine_pairing": {{
                "wine": "Nome del vino consigliato",
                "description": "Breve spiegazione del perché questo vino si abbina perfettamente"
            }}
        }},
        {{
            "title": "Nome creativo della ricetta 2",
            "prep_time": 30,
            "difficulty": "media",
            "servings": 4,
            "calories": 520,
            "description": "Descrizione ricetta 2",
            "ingredients": [...],
            "steps": [...],
            "bon_appetit": "Frase elegante",
            "wine_pairing": {{"wine": "...", "description": "..."}}
        }},
        {{
            "title": "Nome creativo della ricetta 3",
            "prep_time": 40,
            "difficulty": "difficile",
            "servings": 4,
            "calories": 680,
            "description": "Descrizione ricetta 3",
            "ingredients": [...],
            "steps": [...],
            "bon_appetit": "Frase elegante",
            "wine_pairing": {{"wine": "...", "description": "..."}}
        }}
    ]
}}

NOTE IMPORTANTI:
- "calories" rappresenta le kcal TOTALI per porzione. Calcola le calorie in modo realistico.
- "bon_appetit" deve essere una frase elegante, poetica e fantasiosa per augurare buon appetito.
- "wine_pairing" deve suggerire un vino italiano o internazionale che si abbini perfettamente al piatto."""
        else:
            course_names_en = {
                "antipasto": "an appetizer",
                "primo": "a first course/pasta",
                "secondo": "a main course",
                "dessert": "a dessert"
            }
            course_en = course_names_en.get(request.course_type, "a dish")
            
            constraints_en = []
            if request.quick_recipe:
                constraints_en.append("that takes maximum 20 minutes to prepare")
            if request.gourmet:
                constraints_en.append("that is elaborate and gourmet, worthy of a star chef")
            constraint_text_en = " ".join(constraints_en) if constraints_en else ""
            
            prompt = f"""Create EXACTLY 3 different recipes for {course_en} that are creative and tasty {constraint_text_en} using mainly these ingredients: {', '.join(request.ingredients)}.

{dietary_text_en}The 3 recipes must be DIFFERENT from each other: one simpler/classic, one more creative/modern, and one more elaborate/gourmet.

Reply ONLY in valid JSON format with this exact structure:
{{
    "recipes": [
        {{
            "title": "Creative recipe name 1",
            "prep_time": 25,
            "difficulty": "easy|medium|hard",
            "servings": 4,
            "calories": 450,
            "description": "A brief description of why this recipe is special (2-3 sentences)",
            "ingredients": [
                {{"name": "ingredient", "quantity": "quantity"}}
            ],
            "steps": [
                "First detailed step",
                "Second detailed step"
            ],
            "bon_appetit": "An elegant and poetic phrase to wish bon appetit (e.g., 'May this dish bring joy to your table!')",
            "wine_pairing": {{
                "wine": "Recommended wine name",
                "description": "Brief explanation of why this wine pairs perfectly"
            }}
        }},
        {{
            "title": "Creative recipe name 2",
            "prep_time": 30,
            "difficulty": "medium",
            "servings": 4,
            "calories": 520,
            "description": "Recipe 2 description",
            "ingredients": [...],
            "steps": [...],
            "bon_appetit": "Elegant phrase",
            "wine_pairing": {{"wine": "...", "description": "..."}}
        }},
        {{
            "title": "Creative recipe name 3",
            "prep_time": 40,
            "difficulty": "hard",
            "servings": 4,
            "calories": 680,
            "description": "Recipe 3 description",
            "ingredients": [...],
            "steps": [...],
            "bon_appetit": "Elegant phrase",
            "wine_pairing": {{"wine": "...", "description": "..."}}
        }}
    ]
}}

IMPORTANT NOTES:
- "calories" represents the TOTAL kcal per serving. Calculate calories realistically.
- "bon_appetit" must be an elegant, poetic and imaginative phrase to wish bon appetit.
- "wine_pairing" should suggest an Italian or international wine that pairs perfectly with the dish."""
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_id,
            system_message="Sei uno chef italiano esperto e creativo. Crei ricette deliziose e innovative. Rispondi sempre in formato JSON valido con esattamente 3 ricette diverse."
        ).with_model("gemini", "gemini-2.5-flash")
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        # Parse JSON response
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            recipes = result.get("recipes", [])
            if recipes:
                return {"recipes": recipes}
            else:
                # Fallback: maybe it's a single recipe
                return {"recipes": [result]}
        else:
            raise HTTPException(status_code=500, detail="Formato risposta non valido")
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail="Errore nel parsing della ricetta")
    except Exception as e:
        print(f"Error generating recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nella generazione della ricetta: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
