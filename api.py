from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HuggingFace Inference API token ──────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_NEW_TOKEN_HERE")
CLIP_API = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

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

# ── Real CLIP via HuggingFace Inference API ───────────────────────────────────
async def get_clip_score(image_bytes: bytes, product_name: str) -> float:
    labels = [
        f"a genuine {product_name} product",
        f"a fake or counterfeit {product_name} product",
        f"an unrelated product that is not {product_name}",
    ]

    payload = {
        "inputs": {
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "candidate_labels": labels,
        }
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(CLIP_API, json=payload, headers=headers)
            result = response.json()

        # result is a list like:
        # [{"label": "a genuine ...", "score": 0.82}, ...]
        if isinstance(result, list):
            genuine_score = 50.0  # default
            for item in result:
                if "genuine" in item.get("label", "").lower():
                    genuine_score = round(item["score"] * 100, 2)
                    break
            return genuine_score
        else:
            # API error or model loading — return neutral score
            print("CLIP API error:", result)
            return 55.0

    except Exception as e:
        print("CLIP request failed:", e)
        return 55.0


# ── Main analyze endpoint ─────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    product_name: str = Form(...),
    description: str = Form(...),
    entered_price: float = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    reasons = []
    product_key = product_name.lower().strip()

    # ── 1. Real CLIP Score ──────────────────────────────────────────────────
    clip_score = await get_clip_score(image_bytes, product_name)

    # Extra penalty if image clearly doesn't match product
    # (CLIP will naturally give low genuine score for mismatched image)

    # ── 2. Price Score ──────────────────────────────────────────────────────
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

    # ── 3. Keyword Score ────────────────────────────────────────────────────
    if product_key in product_db:
        keywords = product_db[product_key]["keywords"]
        desc_lower = description.lower()
        matches = [kw for kw in keywords if kw in desc_lower]
        keyword_score = round((len(matches) / len(keywords)) * 100, 2)
        if keyword_score < 50:
            reasons.append("Description missing expected product keywords")
    else:
        keyword_score = 75

    # ── 4. Image-Name mismatch penalty ─────────────────────────────────────
    # If CLIP genuine score is very low, it means image doesn't match product
    if clip_score < 35:
        reasons.append("Product image does not appear to match the stated product name")

    # ── 5. Final weighted score ─────────────────────────────────────────────
    final_score = round(
        (clip_score  * 0.50) +
        (price_score * 0.30) +
        (keyword_score * 0.20),
    2)

    # ── 6. Risk level ───────────────────────────────────────────────────────
    if final_score >= 75:
        risk  = "Low Risk — Likely Genuine"
        level = "low"
    elif final_score >= 50:
        risk  = "Medium Risk — Suspicious"
        level = "medium"
    else:
        risk  = "High Risk — Likely Fake"
        level = "high"

    return {
        "clip_score":    clip_score,
        "price_score":   price_score,
        "keyword_score": keyword_score,
        "final_score":   final_score,
        "risk":          risk,
        "level":         level,
        "reasons":       reasons,
    }
