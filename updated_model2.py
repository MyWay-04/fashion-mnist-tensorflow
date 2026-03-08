# Import libraries
import logging
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modelLogger")

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

# # Define the model ex1
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),   # เพิ่ม layer ตรงนี้
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Train model
model.fit(ds_train, epochs=5)

# Show model summary
model.summary()

# Log summary message
logger.info("Model training completed successfully.")