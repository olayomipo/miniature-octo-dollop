from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, val_data, epochs=5):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
    return history
