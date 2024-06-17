from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model('/content/my_model.h5')

# Load the test image
test_image_path = '/content/image_170.jpg'
test_image = load_img(test_image_path, target_size=(224, 224))

# Convert the test image to a numpy array
test_image = img_to_array(test_image)

# Reshape the test image
test_image = np.expand_dims(test_image, axis=0)

# Normalize the test image
test_image /= 255.0

# Make the prediction
prediction = model.predict(test_image)
print(prediction)

# Print the predicted class
if prediction[0][0] > prediction[0][1]:
    print('The image is predicted to be in class Normal_ROI')
else:
    print('The image is predicted to be in class Glaucoma_ROI')