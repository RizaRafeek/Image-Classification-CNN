# 🏎️ CIFAR-10 Image Classification: CNN Architecture
> **Developing a Convolutional Neural Network (CNN) for multi-class object recognition in low-resolution (32x32) color imagery.**

## 📌 Project Overview
This project implements a custom Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into ten distinct categories. The primary engineering challenge was extracting meaningful spatial features from low-resolution pixels while preventing overfitting in a deep architecture.

## 🛠️ Technical Strategy & Architecture
I designed a Sequential CNN architecture optimized for spatial pattern recognition:

* **Feature Extraction:** Utilized `Conv2D` layers with 32 and 64 filters to scan for hierarchical patterns (edges $\rightarrow$ shapes $\rightarrow$ complex textures).
* **Dimensionality Reduction:** Implemented `MaxPooling2D` to reduce spatial variance and computational load.
* **Regularization Suite:** * **Dropout (0.5):** Applied a high dropout rate to the dense layer to force feature redundancy and prevent co-adaptation.
    * **Early Stopping:** Monitored `val_loss` with a patience of 2 to halt training at the point of optimal generalization.
* **Data Normalization:** Scaled RGB pixel values to a $[0, 1]$ range to ensure stable gradient descent and faster convergence.

## 📈 Performance Analysis
| Metric | Result |
| :--- | :--- |
| **Final Validation Accuracy** | **~66.5%** |
| **Optimization Algorithm** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Hardware Acceleration** | GPU (T4) |

### **Generalization Review**
The Training vs. Validation accuracy curves show high alignment, indicating that the **Regularization Strategy** was successful. While the accuracy is capped, the model shows zero signs of overfitting—a critical requirement for production-ready AI.

## 🚀 Key Engineering Insights
1.  **Spatial vs. Dense:** Unlike standard ANNs, the CNN architecture preserves the spatial relationship between pixels, which is why it outperformed baseline dense models significantly.
2.  **The Overfitting Trade-off:** The high Dropout (0.5) prevented overfitting but likely limited the maximum accuracy. For future iterations, **Data Augmentation** would be a superior alternative to high dropout.
3.  **Early Stopping:** The model converged early (around epoch 8), proving that the feature space was learned efficiently without redundant compute cycles.

## 💻 How to Use
1.  **Model Loading:** Load the pre-trained `cifar10_cnn_model.h5` using `tensorflow.keras.models.load_model`.
2.  **Inference:** Ensure input images are reshaped to `(1, 32, 32, 3)` and normalized to `[0, 1]`.
3.  **Classes:** Refer to `cifar10_classes.json` for index-to-label mapping.
