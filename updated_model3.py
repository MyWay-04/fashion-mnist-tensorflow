# Import libraries
import logging
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# Setup logging for VS Code / terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modelLogger")

# Define, load and configure data
(ds_train, ds_test), info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)

# Define batch size
BATCH_SIZE = 32

# Batch processing (NO normalization for Exercise 3)
ds_train = ds_train.batch(BATCH_SIZE)
ds_test = ds_test.batch(BATCH_SIZE)

# Define the model
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

# Train model
model.fit(ds_train, epochs=5)

# Show model summary
model.summary()

# Print max pixel value to verify no normalization
image_batch, labels_batch = next(iter(ds_train))
t_image_batch, t_labels_batch = next(iter(ds_test))

logger.info("training images max " + str(np.max(image_batch[0])))
logger.info("test images max " + str(np.max(t_image_batch[0])))