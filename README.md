# Cat vs Dog Image Classifier (Transfer Learning)

A beginner machine learning project that classifies images as **cat** or **dog** using transfer learning on **MobileNetV2**, pretrained on ImageNet. Built as a learning project to understand neural networks and computer vision from the ground up.

## What it does

Given a photo, the model outputs a confidence score for "cat" vs "dog," using a pretrained CNN (MobileNetV2) with the final layers replaced and retrained on the [Cats vs Dogs dataset](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs).

## How it works

1. **Transfer learning:** Instead of training a CNN from scratch (which needs millions of images), this loads MobileNetV2's weights from being pretrained on 1.4M general images, freezes the early layers (which already know how to detect edges/shapes/textures), and trains only a small new classification head on this specific task.
2. **Fine-tuning:** After the new head is trained, the top ~30 layers of MobileNetV2 are unfrozen and trained further at a very low learning rate, letting the pretrained features adjust slightly to this specific dataset.
3. **Data augmentation:** Training images are randomly flipped/rotated/zoomed so the model generalizes rather than memorizes.

## Results

- Validation accuracy: `__%` 
- Training/validation accuracy curves: 

## Tech stack

- Python, TensorFlow / Keras
- TensorFlow Datasets (`cats_vs_dogs`)
- Gradio (web demo interface)
- Developed in Google Colab

## Running it

**Train the model** (Colab recommended for free GPU):
```bash
pip install tensorflow tensorflow-datasets matplotlib
python train.py
```
This downloads the dataset, trains the model, and saves it as `cat_dog_classifier.keras`.

**Run the demo:**
```bash
pip install gradio
python app.py
```
Opens a local web page where you can drag and drop a photo and get a live prediction.

## Known limitation: out-of-distribution inputs

This model was trained on exactly two categories: cat and dog. It has no concept of "neither" — feed it a photo of anything else (an apple, a car, a person) and it will still confidently output cat or dog, because its final layer is mathematically forced to choose between those two options.

`app.py` includes a partial mitigation: predictions close to 50/50 confidence are flagged as "low confidence — may not be a cat or dog at all." This doesn't solve the underlying problem, but it avoids presenting an unrelated-object prediction as if it were certain. Properly solving this would require either a third "other" training class or dedicated out-of-distribution detection techniques — a good next step beyond this project's scope.

## Next steps / ideas for extending this

- Add a third "neither" class trained on random non-cat/dog images
- Try a different base model (ResNet50, EfficientNet) and compare accuracy
- Deploy the Gradio demo publicly via Hugging Face Spaces
- Expand to more categories (dog breeds, other animals, etc.)

## Project background

Built as a self-directed introduction to machine learning, going from "what is a neural network" through data preprocessing, transfer learning, fine-tuning, and deployment.
