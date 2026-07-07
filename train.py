"""
Cat vs Dog Image Classifier — Transfer Learning with MobileNetV2
Run this in Google Colab (or locally with TensorFlow installed).

NOTE: Colab's pre-installed tensorflow-datasets and protobuf versions can
drift out of sync, causing confusing errors on import. If you hit errors
loading tensorflow_datasets, run this in its own cell first, then
Runtime -> Restart runtime, then re-run this script:

    !pip install --upgrade --no-deps tensorflow-datasets protobuf
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
(train_ds, val_ds), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[20%:]'],
    with_info=True,
    as_supervised=True,
)
class_names = ds_info.features['label'].names
print("Classes:", class_names)

# ---------------------------------------------------------
# 2. Preprocess: resize every image to the same shape
# ---------------------------------------------------------
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

train_dataset = (train_ds.map(preprocess)
                          .shuffle(1000)
                          .batch(BATCH_SIZE)
                          .prefetch(tf.data.AUTOTUNE))
validation_dataset = (val_ds.map(preprocess)
                             .batch(BATCH_SIZE)
                             .prefetch(tf.data.AUTOTUNE))

# ---------------------------------------------------------
# 3. Data augmentation — only affects training data.
#    Randomly flips/rotates images so the model sees more
#    variety and doesn't just memorize exact training photos.
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ---------------------------------------------------------
# 4. Build the model: pretrained MobileNetV2 + custom head
# ---------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze for the first training phase

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- Phase 1: training the new top layer only ---")
history = model.fit(train_dataset, epochs=5, validation_data=validation_dataset)

# ---------------------------------------------------------
# 5. Fine-tuning: unfreeze the top part of MobileNetV2 and
#    train a bit more with a much lower learning rate.
#    This lets the pretrained features adjust slightly to
#    your specific dataset, usually improving accuracy further.
# ---------------------------------------------------------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # unfreeze only the last ~30 layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n--- Phase 2: fine-tuning the top layers of MobileNetV2 ---")
fine_tune_epochs = 5
history_fine = model.fit(train_dataset,
                          epochs=fine_tune_epochs,
                          validation_data=validation_dataset)

# ---------------------------------------------------------
# 6. Plot accuracy/loss so you can see training progress
# ---------------------------------------------------------
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=5, color='gray', linestyle='--', label='Start of fine-tuning')
plt.legend()
plt.title('Accuracy over training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# ---------------------------------------------------------
# 7. Save the trained model so you don't have to retrain
#    every time you want to use it (e.g. in the Gradio app).
# ---------------------------------------------------------
model.save('cat_dog_classifier.keras')
print("\nModel saved as cat_dog_classifier.keras")
