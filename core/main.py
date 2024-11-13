import tensorflow as tf
from data_preprocessing import load_data, preprocess_image, prepare_custom_data
from model import create_model, train_model
from predict import predict_image
from core.correct_and_retrain import correct_and_retrain

# Load and preprocess dataset
train_data, val_data, test_data = load_data()
train_data = train_data.map(preprocess_image)
val_data = val_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

# Load custom data
custom_data_path = 'path/to/custom_dataset'
train_custom_data, val_custom_data = prepare_custom_data(custom_data_path)

# Define and train the model
model = create_model(num_classes=train_custom_data.num_classes)
train_model(model, train_custom_data, val_custom_data, epochs=5)

# Save the model
model.save('plant_identifier_model.h5')

# Make a prediction on a new image
class_names = list(train_custom_data.class_indices.keys())
predict_image(model, 'path/to/test_image.jpg', class_names)

# Correct and retrain if necessary
correct_and_retrain(model, custom_data_path, epochs=5)
