from keras.preprocessing import image
from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import load_model
import numpy as np

saved_model = load_model("data/cnn/model.h5", compile=False)
saved_model.make_predict_function()

def main(image):
    img = image.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    prediction = saved_model.predict(img)

    print(prediction)

    # 0 means KTP is detected
    return prediction[0][0] == 0

if __name__ == '__main__':
    main(sys.argv[1])