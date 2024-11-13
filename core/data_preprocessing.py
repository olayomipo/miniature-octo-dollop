import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

# Define image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

def load_data():
    # Load Oxford Flowers dataset (example dataset for testing)
    
    (train_data, val_data, test_data), ds_info = tfds.load(
        'oxford_flowers102',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    return train_data, val_data, test_data

def preprocess_image(image, label):
    # Resize and normalize images
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

def prepare_custom_data(custom_data_path):
    custom_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )
    train_custom_data = custom_datagen.flow_from_directory(
        custom_data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )
    val_custom_data = custom_datagen.flow_from_directory(
        custom_data_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )
    return train_custom_data, val_custom_data
