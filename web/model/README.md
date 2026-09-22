# Model files go here

This folder holds the fine-tuned MobileNetV2 from experiment E9, converted to
TensorFlow.js so it runs inside the farmer's browser.

Produce it in Colab, at the end of the notebook, once E9 has been trained:

```python
transfer.export("maize_e9_savedmodel")
!pip install -q tensorflowjs
!tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model \
    --quantize_float16=* maize_e9_savedmodel tfjs_model
!zip -r tfjs_model.zip tfjs_model
```

Download `tfjs_model.zip`, unzip it, and copy its contents into this folder, so
that the files sit here as:

```
web/model/model.json
web/model/group1-shard1of2.bin
web/model/group1-shard2of2.bin
```

The exact number of `.bin` shards varies. The page loads `model/model.json` and
the shards follow automatically.

The model must stay in step with `web/index.html`, which feeds it 128x128
images scaled into [0, 1], in the class order Blight, Common_Rust,
Gray_Leaf_Spot, Healthy.
