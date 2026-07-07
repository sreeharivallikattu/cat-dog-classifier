"""
Gradio web demo for the Cat vs Dog classifier.
"""

import gradio as gr
import tensorflow as tf
import numpy as np

IMG_SIZE = (160, 160)
CLASS_NAMES = ['cat', 'dog']

model = tf.keras.models.load_model('cat_dog_classifier.keras')

# How close to 0.5 counts as "not confident enough to guess"
UNCERTAINTY_MARGIN = 0.15


def predict(image):
    """
    image: a PIL image handed to us automatically by Gradio.
    Returns a dictionary of {label: confidence} that Gradio's
    Label component knows how to display as a bar chart.
    """
    img = tf.image.resize(np.array(image), IMG_SIZE)
    img = tf.expand_dims(img, axis=0)  # model expects a batch, so add a batch dimension of 1

    prediction = model.predict(img)[0][0]  # a single number between 0 and 1

    dog_confidence = float(prediction)
    cat_confidence = 1 - dog_confidence

    # If the prediction sits close to 0.5, the model is unsure —
    # this is our practical (partial) fix for the "apple -> cat" problem.
    # It won't recognize unrelated objects, but it will at least flag
    # low-confidence guesses instead of stating them as fact.
    if abs(dog_confidence - 0.5) < UNCERTAINTY_MARGIN:
        note = "⚠️ Low confidence — this may not be a cat or dog at all."
    else:
        note = ""

    return {
        "cat": cat_confidence,
        "dog": dog_confidence,
    }, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Note"),
    ],
    title="Cat vs Dog Classifier",
    description=(
        "Upload a photo and the model will predict cat or dog, "
        "using transfer learning on MobileNetV2. "
        "Note: this model only knows cats and dogs — images of anything else "
        "will still get forced into one of those two categories, though low-confidence "
        "predictions will be flagged."
    ),
)

if __name__ == "__main__":
    demo.launch()
