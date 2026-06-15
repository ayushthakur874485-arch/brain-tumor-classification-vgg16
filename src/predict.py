# Move your prediction code here
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

IMAGE_SIZE = 128

class_labels = [
    'glioma',
    'notumor',
    'meningioma',
    'pituitary'
]

model = load_model("model/model.h5")

def predict_image(image_path):

    img = load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    img_array = img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)

    confidence = np.max(prediction)

    result = class_labels[class_index]

    plt.imshow(load_img(image_path))
    plt.axis("off")
    plt.title(
        f"{result} ({confidence*100:.2f}%)"
    )
    plt.show()

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence*100:.2f}%")

# Example
predict_image("sample.jpg")
