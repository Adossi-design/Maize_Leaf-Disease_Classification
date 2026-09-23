# The model the web app runs

`model.json` plus the `.bin` shards are the fine-tuned MobileNetV2 from
experiment E9, converted for TensorFlow.js and quantised to float16, which puts
the download at about 4.5 MB.

## Scores

Trained on this machine against the split recorded in `split.csv`, so the
2,930 / 629 / 629 images are the same ones the notebook used:

| | Accuracy | Macro-F1 |
|---|---|---|
| E8, frozen base | 0.9205 | 0.9005 |
| E9, fine-tuned (this model) | 0.9300 | 0.9102 |

Per class on the 629 test images, for E9:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Blight | 0.88 | 0.90 | 0.89 |
| Common Rust | 0.96 | 0.96 | 0.96 |
| Gray Leaf Spot | 0.80 | 0.77 | 0.79 |
| Healthy | 1.00 | 1.00 | 1.00 |

Gray Leaf Spot remains the weak class: 18 of its 86 test leaves are called
Blight. The app says so whenever it reports either of those two.

## How it was produced

1. Trained with the notebook's recipe: seed 42, 128x128 input scaled to [0, 1],
   batch 32, the same augmentation block, Adam 1e-3 on the frozen base with
   early stopping, then the top 30 layers unfrozen at Adam 1e-5.
2. Rebuilt for inference as `Rescaling(2, -1) -> MobileNetV2 -> pooling ->
   dense`, with the trained weights copied over. At inference the augmentation
   is a no-op and `preprocess_input(x * 255)` is exactly `x * 2 - 1` on [0, 1]
   input, so the two models compute the same thing. They were compared on 48
   real test images: identical labels, largest probability difference 3e-6.
   The rebuild exists because the training graph carries those steps as loose
   ops that the TensorFlow.js converter cannot freeze.
3. Converted with `tensorflowjs_converter --input_format=keras
   --output_format=tfjs_layers_model --quantize_float16="*"`.

Verified in the browser afterwards: the same four example leaves give the same
answers in the page as in Python, within 0.4%.

## Replacing it

`web/index.html` loads `model/model.json` and handles both a layers model and a
graph model. Any replacement must keep the input at 128x128 scaled to [0, 1] and
the class order Blight, Common_Rust, Gray_Leaf_Spot, Healthy.
