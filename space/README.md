---
title: Maize Leaf Diagnostic
emoji: 🌽
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Maize Leaf Diagnostic

Photograph one maize leaf and get a reading: Blight, Common Rust, Gray Leaf Spot
or Healthy. The model is a MobileNetV2 fine-tuned on 4,188 leaf photos from the
[Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset),
and it scored 93.2% accuracy and 0.912 macro-F1 on a held-out test set of 629
images.

Built by Adossi Fred William as part of a machine learning project comparing
traditional machine learning against deep learning on the same data.
Source: https://github.com/Adossi-design/Maize_Leaf-Disease_Classification

## Using the web app

Open the Space, take or upload a close-up photo of one leaf, and press Predict.
The app checks the photo for darkness, blur and whether a leaf is visible at all,
then reports how confident it is:

- 80% or more: very likely
- 55% to 80%: possibly, worth checking a second leaf
- under 55%: no answer given, take another photo

Blight and Gray Leaf Spot look alike, and the model confuses them in about one
in four Gray Leaf Spot cases, so results for those two carry a warning.

## Using the API

    curl -X POST https://<your-space>.hf.space/predict -F "file=@leaf.jpg"

```json
{
  "label": "Gray Leaf Spot",
  "class_id": "Gray_Leaf_Spot",
  "confidence": 0.87,
  "band": "sure",
  "probabilities": {"Blight": 0.11, "Common Rust": 0.01, "Gray Leaf Spot": 0.87, "Healthy": 0.01},
  "green_fraction": 0.62,
  "inference_ms": 91
}
```

`GET /health` reports whether the model loaded, its input shape and its test scores.

## Running it locally

    pip install -r requirements.txt
    uvicorn app:app --reload --port 7860

The model file `maize_e9.keras` must sit next to `app.py`, or `MODEL_PATH` must
point at it.

## Limits

The model knows four things only, and cannot recognise other crops, pests or
drought damage. It learned mostly from close-up photos taken in good light, so
photos taken in a field are less reliable. It is a helper, not a diagnosis:
confirm with an agricultural extension officer before spending money on
treatment.
