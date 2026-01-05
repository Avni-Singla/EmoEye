# EmoEye
Emotion Detection through Eye Images Using Deep Learning

##📌 Overview
EmoEye is a deep learning–based emotion recognition system that classifies six human emotions using eye-region images only, ensuring privacy preservation while maintaining high performance. The project leverages transfer learning with multiple CNN architectures and focuses on robustness, explainability, and real-time inference suitability.

##🎯 Objectives
- Detect human emotions using eye-region images
- Preserve privacy by avoiding full-face analysis
- Compare multiple deep learning architectures
- Achieve reliable performance on unseen data
- Provide model interpretability using Grad-CAM

##😃 Emotion Classes
The system classifies the following six emotions:
1. Anger
2. Disgust
3. Fear
4. Happiness
5. Sadness
6. Surprise

##🧠 Models Implemented
The project evaluates and compares the following transfer learning models:
- ResNet50
- VGG16
- MobileNetV2
- DenseNet121
Each model is fine-tuned on the EmoEye dataset and evaluated using standard classification metrics.

##📂 Repository Structure
```
EmoEye/
│
├── EmoEye_Dataset.zip        # Eye-region emotion dataset
│
├── ResNet50.ipynb            # ResNet50 training & evaluation
├── VGG16.ipynb               # VGG16 training & evaluation
├── MobileNetV2.ipynb         # MobileNetV2 training & evaluation
├── denseNet121.ipynb         # DenseNet121 training & evaluation
│
└── README.md                 # Project documentation
```

##⚙️ Methodology
1. Dataset Preparation
- Eye-region images organized into emotion-wise folders
- Data preprocessing and normalization
- Train–validation split
2. Model Training
- Pretrained ImageNet weights
- Fine-tuning of higher layers
- Optimization using Adam optimizer
- Categorical Cross-Entropy loss
3. Evaluation
- Accuracy
- Precision, Recall, F1-score
- Performance comparison across models
4. Explainability
- Grad-CAM visualizations to highlight salient eye regions
- Model behavior analysis on unseen samples

##📊 Results Summary
- Best Accuracy: 76.81%
- Best F1-Score: 0.7413
- Inference Time: Sub-second per image
- Robustness: Minimal misclassification on unseen test samples
- Deployment Stability: VGG16 demonstrated consistent predictions

##🛠️ Tech Stack
- Programming Language: Python
- Deep Learning: TensorFlow, Keras
- Models: CNNs, Transfer Learning
- Visualization: Matplotlib, Seaborn
- Explainability: Grad-CAM
- Environment: Google Colab Notebook

##🚀 How to Run
### 1. Clone the repository
```bash
!git clone https://github.com/your-username/EmoEye.git
%cd EmoEye
```
2. Download the dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mdnymurrahmanshuvo/eye-emotion-dataset-diu
Upload the extracted dataset to your Colab workspace or mount Google Drive.

3. Open and run a model notebook
Open any of the following notebooks in Google Colab and run all cells:
- ResNet50.ipynb
- VGG16.ipynb
- MobileNetV2.ipynb
- denseNet121.ipynb
Run all cells to train and evaluate the model

##🔮 Future Work
- Cross-dataset generalization
- Real-time webcam inference
- Emotion-aware conversational AI integration
- Lightweight deployment using TensorFlow Lite
- Attention-based and transformer models

##📊 Results and Discussion
The EmoEye system was evaluated using multiple deep transfer learning models to assess accuracy, robustness, and generalization capability for eye-region–based emotion recognition. Extensive experiments were conducted using standard classification and reliability metrics.
Key findings show that while deeper models achieved higher numerical accuracy, models with simpler architectures demonstrated stronger generalization on unseen data.

## 📈 Model Performance
| Model         | Accuracy | Precision | Recall | F1-Score | Cohen’s Kappa |
|--------------|----------|-----------|--------|----------|---------------|
| ResNet50 (50 epochs) | **0.7681** | 0.7862 | 0.7681 | 0.7682 | **0.7176** |
| ResNet50 (40 epochs) | 0.7101 | 0.7211 | 0.7101 | 0.7000 | 0.6499 |
| VGG16        | 0.7319 | 0.7711 | 0.7319 | **0.7413** | 0.6766 |
| MobileNetV2  | 0.7101 | 0.7101 | 0.7101 | 0.7012 | 0.6486 |
| DenseNet121  | 0.5870 | 0.6303 | 0.5870 | 0.5916 | 0.5024 |

##🧠 Key Observations
- ResNet50 achieved the highest accuracy (76.81%), benefiting from deep residual learning, but showed signs of overfitting and reduced stability on unseen samples.
- VGG16 demonstrated the best balance between accuracy and generalization, achieving the highest F1-score (0.7413) and strong Cohen’s Kappa, making it the most reliable model for deployment.
- MobileNetV2 delivered competitive results with significantly lower computational cost, indicating suitability for real-time and embedded systems.
- DenseNet121 underperformed due to over-parameterization relative to dataset size.

## 🧪 Generalization on Unseen Images
- VGG16 misclassified only **1 out of 6 unseen samples**
- ResNet50 misclassified **3 unseen samples**
- MobileNetV2 showed stable but moderate performance
- DenseNet121 lacked consistent predictions

## 🔍 Explainability (Grad-CAM)
- Attention concentrated on eyelids, eyebrows, and periocular muscles
- No reliance on background artifacts
- Supports trustworthy deployment in healthcare scenarios

## 🏁 Final Outcome
- Privacy-preserving emotion recognition using only eye-region images
- Strong performance despite limited dataset size
- VGG16 identified as the most reliable deployment candidate
- Grad-CAM ensured transparent and interpretable decision-making
