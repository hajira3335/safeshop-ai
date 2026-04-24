import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

st.set_page_config(page_title="SafeShop AI", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

product_db = {
    "lays chips": {"min_price": 10, "max_price": 20, "keywords": ["chips", "snack", "potato", "lays"]},
    "bisleri water": {"min_price": 15, "max_price": 25, "keywords": ["water", "mineral", "bottle", "bisleri"]},
    "coca cola": {"min_price": 20, "max_price": 40, "keywords": ["cola", "drink", "soda", "cold drink"]},
    "maggi noodles": {"min_price": 12, "max_price": 25, "keywords": ["noodles", "maggi", "instant", "masala"]},
    "parle g biscuit": {"min_price": 5, "max_price": 15, "keywords": ["biscuit", "parle", "glucose", "cookie"]},
    "dettol soap": {"min_price": 30, "max_price": 60, "keywords": ["soap", "dettol", "antiseptic", "germ"]},
    "colgate toothpaste": {"min_price": 40, "max_price": 100, "keywords": ["toothpaste", "colgate", "dental", "mint"]},
    "amul butter": {"min_price": 50, "max_price": 60, "keywords": ["butter", "amul", "dairy", "milk"]},
    "lifebuoy soap": {"min_price": 25, "max_price": 50, "keywords": ["soap", "lifebuoy", "germ", "hygiene"]},
    "tata salt": {"min_price": 20, "max_price": 30, "keywords": ["salt", "tata", "iodized", "sodium"]},
}

st.title("🛡️ SafeShop AI")
st.subheader("Multi-Factor Product Authenticity Checker")
st.markdown("---")

product_name = st.text_input("📦 Product Name (e.g. Lays chips)")
description = st.text_area("📝 Product Description")
entered_price = st.number_input("💰 Enter Price (₹)", min_value=1)
uploaded_image = st.file_uploader("📤 Upload Product Image", type=["jpg", "jpeg", "png"])

if st.button("🔍 Check Authenticity"):
    if not uploaded_image or not product_name:
        st.warning("Please upload an image and enter product name.")
    else:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Product", width=300)

        reasons = []

        labels = [
            f"a genuine {product_name} product",
            f"a fake or counterfeit {product_name} product"
        ]
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        clip_score = round(float(probs[0][0]) * 100, 2)

        product_key = product_name.lower().strip()
        if product_key in product_db:
            db = product_db[product_key]
            if entered_price < db["min_price"] * 0.6:
                price_score = 20
                reasons.append("⚠️ Price is significantly lower than expected")
            elif entered_price < db["min_price"]:
                price_score = 70
                reasons.append("⚠️ Price is slightly below expected range")
            else:
                price_score = 100
            keywords = product_db[product_key]["keywords"]
            desc_lower = description.lower()
            matches = [kw for kw in keywords if kw in desc_lower]
            keyword_score = (len(matches) / len(keywords)) * 100
            if keyword_score < 50:
                reasons.append("⚠️ Description missing expected product keywords")
        else:
            price_score = 80
            keyword_score = 80
            st.info("ℹ️ Product not in database — using general AI analysis only.")

        final_score = round(
            (clip_score * 0.5) +
            (price_score * 0.3) +
            (keyword_score * 0.2),
        2)

        if final_score >= 75:
            risk = "✅ Low Risk — Likely Genuine"
            color = "green"
        elif final_score >= 50:
            risk = "⚠️ Medium Risk — Suspicious"
            color = "orange"
        else:
            risk = "❌ High Risk — Likely Fake"
            color = "red"

        st.markdown("---")
        st.markdown(f"### Authenticity Score: `{final_score}%`")
        st.markdown(f"### Risk Level: :{color}[{risk}]")

        if reasons:
            st.markdown("**Reasons for suspicion:**")
            for r in reasons:
                st.write(r)
        else:
            st.success("No major red flags detected.")
