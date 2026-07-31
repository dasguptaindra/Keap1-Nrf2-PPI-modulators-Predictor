#  Keap1-Nrf2-PPI inhibitors Predictor

Welcome to the **Keap1-Nrf2-PPI inhibitors Predictor**, a machine learning-based web application developed to assist researchers, medicinal chemists, and drug discovery scientists in predicting the activity of **Keap1–Nrf2 protein–protein interaction (PPI) inhibitors** from molecular structures.

---

## 🚀 Launch the Web App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://keap1-nrf2-ppi-modulators-predictor.streamlit.app/)

Click the Streamlit badge above to launch the application.

---

## ✨ Features

- 🧠 **AI-Powered Prediction**
  - Predict the activity of Keap1–Nrf2 PPI inhibitors using a trained ML model.

- ✏️ **Flexible Input**
  - Draw molecules using the integrated **Ketcher** molecular editor.
  - Or directly enter **SMILES** strings.

- 📂 **Batch Prediction**
  - Upload CSV or Excel files containing multiple compounds for high-throughput virtual screening.

- 📊 **Explainable AI**
  - Interpret individual predictions using **SHAP Waterfall plots**, highlighting fingerprint contributions.

- 🎯 **Applicability Domain Analysis**
  - Assess prediction reliability using leverage approach.

- 🌍 **Accessible Anywhere**
  - Fully deployed online using **Streamlit**.


## 🔬 Background

The **Keap1–Nrf2 signaling pathway** plays a critical role in regulating oxidative stress and cellular defense mechanisms. Small molecules capable of disrupting the **Keap1–Nrf2 protein–protein interaction** have emerged as promising therapeutic candidates for numerous diseases.

This web application employs a validated machine learning model to facilitate:

- Virtual screening
- Hit prioritization
- Early-stage drug discovery
- Lead optimization

---

## 📁 Repository Structure

```
.
├── app.py
├── train_model.py
├── descriptor_utils.py
├── requirements.txt
├── data
│   ├── Train_keap.xlsx
│   └── Test_keap.xlsx
└── artifacts/
    ├── model.pkl
    ├── scaler.pkl
    ├── descriptors.pkl
    └── ...
```

The **artifacts** directory is automatically generated after model training.

---

## 📝 Input Options

### Single Compound Prediction

- Draw a molecule using the embedded **Ketcher** editor.
- Or paste a **SMILES** string.

---

### Batch Prediction

Upload a **CSV** or **Excel (.xlsx)** file containing a column named:

```
SMILES
```

For each compound, the application reports:

- Predicted activity
- Average Top-5 Tanimoto similarity
- Applicability Domain threshold
- Applicability Domain status (Within/Outside)

---

## 🧬 Applicability Domain (AD)

Prediction reliability is evaluated using **leverage approach**.

## ⭐ Citation

If you use this application in your research, please cite the corresponding publication (to be added).

## 🤝 Support

If you find this project useful:

⭐ Star this repository


📢 Share it with fellow researchers and students
