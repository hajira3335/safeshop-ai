from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load CLIP locally (no HF token / no remote API needed) ────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# ── Product database ──────────────────────────────────────────────────────────
product_db = {
    "lays chips":         {"min_price": 10,  "max_price": 20,  "keywords": ["chips", "snack", "potato", "lays"]},
    "bisleri water":      {"min_price": 15,  "max_price": 25,  "keywords": ["water", "mineral", "bottle", "bisleri"]},
    "coca cola":          {"min_price": 20,  "max_price": 40,  "keywords": ["cola", "drink", "soda", "cold drink"]},
    "maggi noodles":      {"min_price": 12,  "max_price": 25,  "keywords": ["noodles", "maggi", "instant", "masala"]},
    "parle g biscuit":    {"min_price": 5,   "max_price": 15,  "keywords": ["biscuit", "parle", "glucose", "cookie"]},
    "dettol soap":        {"min_price": 30,  "max_price": 60,  "keywords": ["soap", "dettol", "antiseptic", "germ"]},
    "colgate toothpaste": {"min_price": 40,  "max_price": 100, "keywords": ["toothpaste", "colgate", "dental", "mint"]},
    "amul butter":        {"min_price": 50,  "max_price": 60,  "keywords": ["butter", "amul", "dairy", "milk"]},
    "lifebuoy soap":      {"min_price": 25,  "max_price": 50,  "keywords": ["soap", "lifebuoy", "germ", "hygiene"]},
    "tata salt":          {"min_price": 20,  "max_price": 30,  "keywords": ["salt", "tata", "iodized", "sodium"]},
    "iphone":             {"min_price": 50000, "max_price": 150000, "keywords": ["iphone", "apple", "smartphone", "ios"]},
    "samsung phone":      {"min_price": 8000,  "max_price": 100000, "keywords": ["samsung", "android", "smartphone", "galaxy"]},
    "boat earphones":     {"min_price": 500,   "max_price": 5000,   "keywords": ["earphones", "boat", "audio", "bass"]},
    "red bull":           {"min_price": 100,   "max_price": 150,    "keywords": ["energy", "drink", "red bull", "bull"]},
    "cadbury dairy milk": {"min_price": 20,    "max_price": 200,    "keywords": ["chocolate", "cadbury", "dairy", "milk"]},
}

# Generic "wrong category" labels so mismatched images get a low score
GENERIC_DISTRACTORS = [
    "shoes", "a handbag", "a wrist watch", "a t-shirt", "a laptop",
    "a cosmetics bottle", "a toy", "a book", "furniture", "a car",
]


# ── Vision score: does the image match the CLAIMED product? ───────────────────
def get_clip_score(image_bytes: bytes, product_name: str) -> float:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    claimed = product_name.strip().lower()

    others = [k for k in product_db if k != claimed]
    labels = [claimed] + others + GENERIC_DISTRACTORS
    prompts = [f"a photo of {label}" for label in labels]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image  # shape [1, num_labels]
    probs = logits.softmax(dim=1)[0]

    # index 0 = the product the seller claims this listing is
    return round(float(probs[0]) * 100, 2)


# ── Main analyze endpoint ─────────────────────────────────────────────────────
@app.post("/analyze")
def analyze(
    product_name: str = Form(...),
    description: str = Form(...),
    entered_price: float = Form(...),
    image: UploadFile = File(...),
):
    image_bytes = image.file.read()
    reasons = []
    product_key = product_name.lower().strip()

    # 1. Vision score
    clip_score = get_clip_score(image_bytes, product_name)

    # 2. Price score
    if product_key in product_db:
        db = product_db[product_key]
        if entered_price < db["min_price"] * 0.5:
            price_score = 10
            reasons.append("Price is extremely low — highly suspicious")
        elif entered_price < db["min_price"] * 0.75:
            price_score = 30
            reasons.append("Price is significantly below expected range")
        elif entered_price < db["min_price"]:
            price_score = 70
            reasons.append("Price is slightly below expected range")
        else:
            price_score = 100
    else:
        price_score = 75
        reasons.append("Product not in database — general analysis only")

    # 3. Keyword score
    if product_key in product_db:
        keywords = product_db[product_key]["keywords"]
        desc_lower = description.lower()
        matches = [kw for kw in keywords if kw in desc_lower]
        keyword_score = round((len(matches) / len(keywords)) * 100, 2)
        if keyword_score < 50:
            reasons.append("Description missing expected product keywords")
    else:
        keyword_score = 75

    # 4. Image-name mismatch flag
    mismatch = clip_score < 35
    if mismatch:
        reasons.append("Product image does not appear to match the stated product name")

    # 5. Final weighted score
    final_score = round(
        (clip_score * 0.50) + (price_score * 0.30) + (keyword_score * 0.20), 2
    )
    if mismatch:
        final_score = min(final_score, 45.0)  # a wrong image can never look "medium risk"

    # 6. Risk level
    if final_score >= 75:
        risk, level = "Low Risk — Likely Genuine", "low"
    elif final_score >= 50:
        risk, level = "Medium Risk — Suspicious", "medium"
    else:
        risk, level = "High Risk — Likely Fake", "high"

    return {
        "clip_score": clip_score,
        "price_score": price_score,
        "keyword_score": keyword_score,
        "final_score": final_score,
        "risk": risk,
        "level": level,
        "reasons": reasons,
    }
