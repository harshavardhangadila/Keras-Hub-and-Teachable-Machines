## Keras-Hub

## 📝 Text Classification Pipelines Summary

This repository explores four progressively complex natural language processing (NLP) pipelines, ranging from beginner-level models to expert architectures that incorporate multiple inputs and outputs. Each model is designed to highlight different techniques in deep learning for text classification using popular datasets and frameworks like TensorFlow, Keras, and Hugging Face Transformers.

- **Basic Level**: Implements a vanilla LSTM model using Keras with padded sequences. It performs binary sentiment analysis on IMDb reviews but struggles due to lack of preprocessing and parameter tuning.
  
- **Intermediate Level**: Fine-tunes a DistilBERT model on the AG News dataset for multi-class classification. This pipeline showcases transfer learning with Transformer models and demonstrates modest improvements even with minimal epochs.

- **Advanced Level**: Builds a dual-output model using RoBERTa for simultaneously predicting sentiment and review length category. This setup is useful for multi-task learning scenarios and shows significantly improved performance on auxiliary features.

- **Expert Level**: Combines text input and auxiliary metadata (text length) into a dual-input architecture with BiLSTM layers. This model aims to simulate production-grade pipelines that leverage both textual and structured features for enhanced performance and explainability.

These examples serve as a practical guide for building and comparing various levels of text classification architectures. They are ideal for students, practitioners, and researchers aiming to understand the scalability and complexity of modern NLP systems.


| Level        | Task Type                   | Model & Technique                                      | Dataset Used         |
|--------------|-----------------------------|---------------------------------------------------------|----------------------|
| Basic        | Binary Sentiment            | Tokenizer + LSTM (Keras Sequential)                    | IMDb (Hugging Face)  |
| Intermediate | News Category Classification| DistilBERT + Fine-Tuning (Hugging Face Transformers)   | AG News (Hugging Face) |
| Advanced     | Dual-label: Sentiment + Length| RoBERTa + Multi-output Model                          | IMDb (sampled)       |
| Expert       | Dual-input: Text + Metadata | Bidirectional LSTM + Auxiliary Feature (Text Length)   | IMDb                 |

## Multi-Level Image Classification Summary

This project demonstrates a progression from simple to advanced image classification using TensorFlow and Keras. It includes:

- **Simple Classification** using pretrained ResNet50 from Keras Hub on CIFAR-10.
- **Intermediate Classification** using U-Net for semantic segmentation on Oxford-IIIT Pets.
- **Advanced Classification** with a custom CNN trained from scratch on CIFAR-10.
- **Expert Classification** with a multi-input model combining MobileNetV2 (image) and statistical meta-features (brightness/contrast) on the TF Flowers dataset.

---

## 📊 Model Comparison Table

| Level        | Dataset           | Model Architecture                     | Features Used                     |
|--------------|-------------------|----------------------------------------|-----------------------------------|
| Beginner     | CIFAR-10          | ResNet50 (Transfer Learning)           | RGB image                          |
| Intermediate | Oxford-IIIT Pet   | U-Net (Semantic Segmentation)          | RGB image + segmentation masks     |
| Advanced     | CIFAR-10          | Custom CNN (Conv + BN + Dropout)       | RGB image                          |
| Expert       | TF Flowers        | MobileNetV2 + Meta-features (2 inputs) | RGB image + brightness & contrast |

---





## Teachable Machines Project

This project demonstrates three distinct machine learning pipelines using **TensorFlow**, **TensorFlow Hub**, and **scikit-learn**, built around different types of input modalities:

- 📸 **Image-based Object Classification** using Teachable Machine exported models.
- 🎥 **Video Frame-based Pose Classification** using MoveNet for pose detection + Logistic Regression.
- 📝 **Text-based Sentiment Classification** using Universal Sentence Encoder (USE) embeddings + Logistic Regression.

---

## 📊 Task-wise Summary Table

| Task                   | Technique                                                |
|------------------------|----------------------------------------------------------|
| 🖼️ Object Classification     | Teachable Machine + TensorFlow                            |
| 🕺 Pose Classification       | MoveNet (TF Hub) + Logistic Regression                    | 
| ✍️ Text Classification       | Universal Sentence Encoder (TF Hub) + Logistic Regression |

---

Youtube: [Keras-Hub-and-Teachable-Machines](https://www.youtube.com/playlist?list=PLCGwaUpxPWO2oLaUWohq5KiRGBYHE9-Kj)

