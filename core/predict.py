import tensorflow as tf
import numpy as np

def predict_image(model, image_path, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    print(f"Predicted Class: {class_names[predicted_class]}")
    return predicted_class
