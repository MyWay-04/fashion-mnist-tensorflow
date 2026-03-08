import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import logging

(ds_train, ds_test), info = tfds.load(
    'fashion_mnist',
    split=['train','test'],
    with_info=True,
    as_supervised=True
)

image_batch, labels_batch = next(iter(ds_train))
print("Before normalization ->", np.min(image_batch[0]), np.max(image_batch[0]))

BATCH_SIZE = 32

ds_train = ds_train.map(
    lambda x,y: (tf.cast(x, tf.float32)/255.0, y)
).batch(BATCH_SIZE)

ds_test = ds_test.map(
    lambda x,y: (tf.cast(x, tf.float32)/255.0, y)
).batch(BATCH_SIZE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(ds_train, epochs=5)

model.evaluate(ds_test)

# Save the model
model.save("saved_model.keras")

# Load the saved model
loaded_model = tf.keras.models.load_model("saved_model.keras")

# Show model summary
print("\nLoaded model summary:")
loaded_model.summary()

# Save the model again with another name
model.save("my_model.keras")

# Load the second model
loaded_model2 = tf.keras.models.load_model("my_model.keras")

# Show summary again
print("\nSecond loaded model summary:")
loaded_model2.summary()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("callbackLogger")

# Define Callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('sparse_categorical_accuracy') > 0.84:
            logger.info("Reached 84% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = MyCallback()

# Load dataset
(ds_train, ds_test), info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)

# Define batch size
BATCH_SIZE = 32

# Normalize dataset
ds_train = ds_train.map(
    lambda x, y: (tf.cast(x, tf.float32)/255.0, y)
).batch(BATCH_SIZE)

ds_test = ds_test.map(
    lambda x, y: (tf.cast(x, tf.float32)/255.0, y)
).batch(BATCH_SIZE)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Train model with callback
model.fit(ds_train, epochs=5, callbacks=[callbacks])