# 🧠 Pretrained VGG16 Image Classifier on ImageNet

This project demonstrates how to use the **VGG16 Convolutional Neural Network** pretrained on **ImageNet** to classify images using Keras and TensorFlow in Google Colab.

---

## 🚀 Project Highlights

- Uses **VGG16** pretrained model from `keras.applications`
- Loads and preprocesses custom image input (e.g., an elephant image)
- Performs **top-3 classification predictions**
- Demonstrates end-to-end pipeline:
  - Model loading
  - Image loading
  - Preprocessing
  - Inference
  - Result decoding and visualization

---

## 🧾 Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab or Jupyter

---

## 📦 How It Works

```python
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
```

1. **Model Setup**:
   ```python
   model = VGG16(weights='imagenet')
   ```

2. **Image Load & Preprocess**:
   ```python
   img = load_img('/content/drive/My Drive/Dataset/image_test.jpg', target_size=(224, 224))
   x = img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)
   ```

3. **Prediction**:
   ```python
   preds = model.predict(x)
   decode_predictions(preds, top=3)[0]
   ```

---

## 🎯 Sample Output

```text
Predicted:
[
  ('n02504458', 'African_elephant', 0.90),
  ('n01871265', 'tusker', 0.067),
  ('n02504013', 'Indian_elephant', 0.015)
]
```

---

## 🔍 Notes

- **Input Image Size**: Must be resized to `(224, 224)` to match VGG16 input requirements.
- **Preprocessing**: `preprocess_input()` scales pixel values and centers them to match VGG training input distribution.
- **Weights**: All model weights are preloaded from ImageNet.
- **No Fine-Tuning**: This project only does inference using frozen weights.

---

## 📌 About VGG16

- 16 weight layers (13 conv + 3 dense)
- Used widely for transfer learning
- Simple, deep architecture with small 3×3 kernels

---

## 📂 File Structure

```
├── vgg16_inference_colab.ipynb
├── image_test.jpg
├── README.md
```

---

## 🧠 Credits

- Model: [VGG16 paper (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- Framework: [Keras Applications Documentation](https://keras.io/api/applications/vgg/)
