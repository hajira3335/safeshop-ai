from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP model once when server starts
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model loaded!")

product_db = {
    "lays chips": {"min_price": 10, "max_price": 20, "keywords": ["chips", "snack", "potato", "lays"]},
    "bisleri water": {"min_price": 15, "max_price": 25, "keywords": ["water", "mineral", "bottle", "bisleri"]},
    "coca cola": {"min_price": 20, "max_price": 40, "keywords": ["cola", "drink", "soda", "cold drink"]},
}

@app.post("/analyze")
async def analyze(
    product_name: str = Form(...),
    description: str = Form(...),
    entered_price: float = Form(...),
    image: UploadFile = File(...)
):
    # Read image
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))

    reasons = []

    # CLIP Score
    labels = [
        f"a genuine {product_name} product",
        f"a fake or counterfeit {product_name} product"
    ]
    inputs = processor(text=labels, images=pil_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    clip_score = round(float(probs[0][0]) * 100, 2)

    # Price Score
    product_key = product_name.lower().strip()
    if product_key in product_db:
        db = product_db[product_key]
        if entered_price < db["min_price"] * 0.6:
            price_score = 20
            reasons.append("Price is significantly lower than expected")
        elif entered_price < db["min_price"]:
            price_score = 70
            reasons.append("Price is slightly below expected range")
        else:
            price_score = 100
    else:
        price_score = 80

    # Keyword Score
    if product_key in product_db:
        keywords = product_db[product_key]["keywords"]
        matches = [kw for kw in keywords if kw in description.lower()]
        keyword_score = (len(matches) / len(keywords)) * 100
        if keyword_score < 50:
            reasons.append("Description missing expected product keywords")
    else:
        keyword_score = 80

    # Final Score
    final_score = round((clip_score * 0.5) + (price_score * 0.3) + (keyword_score * 0.2), 2)

    # Risk Level
    if final_score >= 75:
        risk = "Low Risk — Likely Genuine"
        level = "low"
    elif final_score >= 50:
        risk = "Medium Risk — Suspicious"
        level = "medium"
    else:
        risk = "High Risk — Likely Fake"
        level = "high"

    return {
        "clip_score": clip_score,
        "price_score": price_score,
        "keyword_score": keyword_score,
        "final_score": final_score,
        "risk": risk,
        "level": level,
        "reasons": reasons
    }
