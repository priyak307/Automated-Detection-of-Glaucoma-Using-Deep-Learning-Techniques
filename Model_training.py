import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load the VGG19 model without the top classification layer
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add LSTM layers
model = Sequential()
model.add(TimeDistributed(vgg19, input_shape=(10, 224, 224, 3)))  # Assuming 10 frames in the sequence
model.add(TimeDistributed(Flatten()))
model.add(LSTM(256, return_sequences=False))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Data augmentation and generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_path = '/content/drive/MyDrive/dataset/train/'
valid_path = '/content/drive/MyDrive/dataset/validation/'

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    classes=['Normal_ROI', 'Glaucoma_ROI'],
    batch_size=60,
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    classes=['Normal_ROI', 'Glaucoma_ROI'],
    batch_size=30)

history = model.fit(
    train_generator,
    steps_per_epoch=5,
    validation_data=validation_generator,
    validation_steps=20,
    epochs=20,
    verbose=2)

# Save the trained model
model.save('/content/drive/MyDrive/my_model.h5')