import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import os

# Path to dataset
data_dir = 'pediatric'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'), target_size=(224, 224), batch_size=16, class_mode='binary')

val_gen = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'), target_size=(224, 224), batch_size=16, class_mode='binary')

# Build EfficientNetB0 model
model = Sequential([
    EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    Flatten(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save the model
model.save('pneumonia_model.h5')
print("Model training complete and saved!")
