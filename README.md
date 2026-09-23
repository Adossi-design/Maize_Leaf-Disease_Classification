# Maize Leaf Disease Classification: Traditional ML vs Deep Learning

This is my machine learning summative project. I built an end-to-end pipeline that
classifies maize (corn) leaf images into four categories and compares a traditional
machine learning approach (Scikit-learn) against deep learning approaches built with
TensorFlow.

## Problem

Maize is a major staple crop in East Africa, including Rwanda, so leaf diseases that
go undetected can cause real losses for farmers. The goal of this project is to see
how well different models can tell maize leaf diseases apart from photos, and to
understand where they fail and why. I treat it as a 4-class image classification
problem.

## Dataset

I used the "Corn or Maize Leaf Disease Dataset" from Kaggle (by Smaranjit Ghose). It
has 4,188 images across four classes:

| Class | Images |
|-------|--------|
| Common Rust | 1,306 |
| Healthy | 1,162 |
| Blight | 1,146 |
| Gray Leaf Spot | 574 |

The dataset is curated from PlantVillage and PlantDoc. It is imbalanced (Gray Leaf
Spot is the smallest class), which I deal with directly in my experiments.

Download link: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

The image files are **not** included in this repo (too large). See `data/README.md`
for how to download and place them.

## Approaches

- **Traditional ML (Scikit-learn):** handcrafted features (HSV color histograms + HOG)
  fed into Logistic Regression, Random Forest, and SVM.
- **Deep learning (TensorFlow):** a CNN built with the Sequential API, a transfer
  learning model (MobileNetV2) built with the Functional API, and a `tf.data` input
  pipeline with augmentation, caching, and prefetching.

## Repository structure

```
maize-leaf-disease-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/        
├── data/            
├── report/          
├── figures/         
└── models/          
```

## How to run

1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset (see `data/README.md`) and place it in the `data/` folder.
4. Open `notebook/maize_disease_classification.ipynb` and run top to bottom.

The notebook sets random seeds so the results are reproducible.

## Results summary

Test set results:

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Logistic Regression | 0.852 | 0.815 |
| Random Forest | 0.867 | 0.795 |
| SVM (RBF, tuned) | 0.881 | 0.855 |
| CNN (Sequential, best of E4 to E7) | 0.898 | 0.886 |
| MobileNetV2 (Functional, fine-tuned) | 0.932 | 0.912 |

The fine-tuned MobileNetV2 was the best model overall. On this run even the plain
baseline CNN beat every classical model, augmentation did not improve results, and the
class-imbalance fixes did not beat the baseline. Gray Leaf Spot stayed the hardest class
for every model, usually confused with Blight.

## The web app

The `web/` folder holds a web app that lets a farmer photograph one maize leaf
and get a reading back, with plain advice for each disease and a warning
whenever Blight and Gray Leaf Spot are in play. The model runs inside the
browser with TensorFlow.js, so photos are never uploaded, and a service worker
keeps the app and the model on the phone, so it still works with no signal. It
is deployed on Vercel from this repository.

The deployed model is a retrained E9, produced on a laptop CPU from the split
recorded in `split.csv`, so it saw exactly the same training, validation and
test images as the notebook run. It scores **0.9300 accuracy and 0.9102
macro-F1** on the 629 test images, next to the notebook's 0.9316 and 0.9117.
The difference is ordinary run-to-run variation. Everything else in this
repository, including the results table above, describes the notebook run.
`web/model/README.md` records the per-class scores and how the conversion was
done.

## Links

- Report: https://docs.google.com/document/d/1b7piDS5D7QGkNwsOnpGy9yrOy5c_IAWAk8_6HFKNihY/edit?usp=sharing

- Demo video: https://docs.google.com/presentation/d/1_piAlzaqfzqzOVnwg2V6BCT9THYPzBV1zBZMvfn9g_g/edit?usp=sharing
