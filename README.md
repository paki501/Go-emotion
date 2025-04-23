# Go-emotion
Absolutely! Here's a comprehensive **GitHub `README.md`-style report** for your **GoEmotions multi-label emotion classification project**, using the same structure as the anomaly detection report.

---

# 🎭 GoEmotions Multi-Label Emotion Classification System

## 📋 Project Overview

The objective of this project is to develop a system capable of detecting and classifying multiple emotional tones (e.g., joy, sadness, anger) in textual data using Google’s **GoEmotions dataset**. The model is built using **BERT**, fine-tuned for **multi-label classification**, making it suitable for customer feedback analysis, social media sentiment, and online behavior monitoring.

---

## 🧹 Dataset Preprocessing Steps

### 📁 Dataset Description:
- **Source**: [GoEmotions by Google](https://github.com/google-research/google-research/tree/master/goemotions)
- **Size**: ~58,000 English Reddit comments
- **Labels**: 27 emotions (12 positive, 11 negative, 4 ambiguous) + 1 neutral

### ✅ Preprocessing Steps:
1. **Data Cleaning**:
   - Removed unnecessary columns like `id`.
   - Cleaned text: Lowercased, removed special characters and redundant spaces.

2. **Label Handling**:
   - Converted multi-label annotations into binary vectors using `MultiLabelBinarizer`.
   - Grouped emotions into four categories for analysis: positive, negative, ambiguous, and neutral.

3. **Handling Imbalanced Data**:
   - Visualized label distribution.
   - Applied stratified train-test split to retain label distribution.
   - Used weighted loss during training to reduce bias toward dominant emotions.

4. **Tokenization**:
   - Used HuggingFace’s `BertTokenizer` to tokenize the comments.
   - Padded and truncated texts to ensure uniform input length.

---

## 🤖 Model Selection and Rationale

### ✅ Model Used: `BERT (bert-base-uncased)`
- Chosen for its strong performance in NLP tasks and support for fine-tuning.
- Adapted with a custom classification head for multi-label outputs using **sigmoid activation**.
- Trained with **Binary Cross-Entropy loss** to support independent emotion predictions.

### 💡 Why BERT?
- Context-aware embeddings for deeper sentiment understanding.
- Pretrained on a large corpus, enabling faster and more accurate fine-tuning.
- Easily extendable and compatible with HuggingFace's ecosystem.

---

## ⚠️ Challenges Faced and Solutions

| Challenge | Solution |
|----------|----------|
| **Highly imbalanced dataset** | Used weighted loss function and stratified split |
| **Multi-label complexity** | Used sigmoid instead of softmax, allowing overlapping emotion detection |
| **Model overfitting** | Early stopping and dropout layers were added |
| **Ambiguous/neutral detection** | Ensured class weights accounted for underrepresented labels |

---

## 📊 Results with Visualizations and Interpretations

### 🧪 Evaluation Metrics:
| Metric          | Value (example)     |
|-----------------|---------------------|
| **Hamming Loss** | 0.073               |
| **Micro F1 Score** | 0.842             |
| **Macro F1 Score** | 0.789             |
| **Accuracy** | ~85% on test set |

---

### 📈 Visualizations:

#### 📌 Correlation Heatmap of Emotions
```python
plt.figure(figsize=(25,20))
sns.heatmap(df.corr(), center=0, annot=True)
```

#### 📊 Emotion Distribution (Positive Emotions)
```python
fig = px.bar(x=positive_col, y=emotion_counts)
fig.update_layout(title = 'Positive Emotions', xaxis_title="Emotion", yaxis_title="Count")
fig.show()
```

#### 🍰 Pie Chart for Emotion Presence
```python
plt.figure(figsize=(5,5))
plt.pie([no_count, yes_count], labels=['No', 'Yes'], autopct='%.0f%%', explode=(0.1, 0.03), shadow=True)
```



