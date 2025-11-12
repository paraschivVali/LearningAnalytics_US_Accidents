# 🚗 Learning Analytics – US Accidents (PySpark)

Proiect realizat în cadrul disciplinei **Learning Analytics**, având ca obiectiv **analiza factorilor meteorologici și temporali** care influențează **severitatea accidentelor rutiere** în SUA, utilizând setul de date *US Accidents (2016–2023)* disponibil pe Kaggle.

---

## 🔍 Scop

Predicția nivelului de severitate al accidentelor rutiere pe baza condițiilor meteorologice (temperatură, vizibilitate, umiditate, precipitații) și a momentului zilei (zi / noapte), folosind modele de clasificare implementate cu **PySpark MLlib**.

---

## ⚙️ Tehnologii

- **Python 3.11**
- **Apache Spark (PySpark MLlib)**
- **pandas**, **matplotlib**
- **PyCharm IDE**

---

## 📊 Modele utilizate

1. **Regresie Logistică (Logistic Regression)**
2. **Arbore de Decizie (Decision Tree Classifier)**
3. **Pădure Aleatorie (Random Forest Classifier)**

---

## 📈 Rezultate

Modelul **Random Forest** a obținut cele mai bune performanțe, oferind un echilibru între acuratețe și complexitate computațională:

| Model | Acuratețe | F1-Score |
|--------|------------|----------|
| Logistic Regression | 0.8455 | 0.7749 |
| Decision Tree | 0.8456 | 0.7755 |
| Random Forest | 0.8456 | 0.7750 |

📊 Graficele comparative ale performanțelor sunt generate automat și salvate în folderul `results/`.

---

## ▶️ Rulare locală

1. Clonează proiectul:
   ```bash
   git clone https://github.com/<user>/LearningAnalytics_US_Accidents.git
   cd LearningAnalytics_US_Accidents
   ```

2. Creează un mediu virtual și instalează dependențele:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Descarcă setul de date din Kaggle și adaugă-l în folderul `data/`:
   👉 [US Accidents (2016–2023) - Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

4. Rulează aplicația:
   ```bash
   python main.py
   ```

---

## 📁 Structura proiectului

```
LearningAnalytics_US_Accidents/
│
├── data/                 # Setul de date (CSV - neîncărcat pe GitHub)
├── results/              # Grafice și rezultate generate
├── main.py               # Script principal
├── requirements.txt      # Dependențe Python
└── README.md             # Documentația proiectului
```

---

## 🧩 Autor

**Paraschiv Valentin**  
Master TIA, anul II – 2025  
Universitatea „Dunarea de Jos” din Galati

---
