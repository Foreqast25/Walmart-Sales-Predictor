#  Walmart Sales Predictor

A full-stack data science and web application that predicts sales for Walmart based on historical patterns, holidays, weather, and economic signals. Built with machine learning, Flask, and a user-friendly interface — this project reduces excess stock and promotes eco-smart retail planning.

▶️ [Click here to watch the demo](https://youtu.be/4HZfc8Bzi2Q?si=A-ULOgTdAksbEXau)


---

##  Project Overview

Walmart stores often struggle with demand volatility due to seasonal shifts, holidays, and unpredictable trends. Our system analyzes key features like holiday indicators, pricing, and weather to forecast product-level sales, helping:
- Prevent **understocking** and lost revenue
- Avoid **overstocking** and waste (eco-smart inventory!)
- Visualize trends for smart decisions

---

##  Tech Stack

| Category         | Tools Used                                     |
|------------------|------------------------------------------------|
| Programming      | Python, JavaScript                             |
| Data Science     | Pandas, NumPy, Scikit-learn, XGBoost            |
| Data Viz         | Matplotlib, Seaborn                            |
| Model Handling   | Pickle, Feature Encoding, Scalers              |
| Frontend         | HTML, CSS (Tailwind-ready), JavaScript         |
| Backend          | Flask (Python)                                 |
| Deployment Ready | Streamlit (optional), Flask                    |
| Version Control  | Git, GitHub                                    |

---

##  Machine Learning Pipeline

- ✅ Data cleaning & preprocessing
- ✅ Feature engineering (holidays, weather, discounts)
- ✅ Encoding + scaling
- ✅ Model selection: **Random Forest Regressor**
- ✅ Evaluation: MSE, RMSE, Actual vs Predicted Plots
- ✅ Model saving: `.pkl` files for deployment

---

##  App Features

-  **Live sales predictions**
-  Actual vs Predicted graphs
-  Calendar View
-  Eco-smart stock suggestions
-  Login & signup functionality
-  Chatbot Assistant (upcoming)
-  Model retrain-ready structure

---

##  Project Structure
walmart-sales-predictor/
├── backend/
│ ├── app.py
│ ├── model/
│ ├── templates/
│ └── static/
├── data/
├── notebooks/
│ └── eda_and_model_training.ipynb
├── README.md
└── requirements.txt


---

##  How to Run Locally

1. Clone the repo
```bash
- git clone https://github.com/Foreqast25/Walmart-Sales-Predictor.git
- cd Walmart-Sales-Predictor
```
2. Create and Activate Virtual Environment
  - Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

  - Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install Required Dependencies
```bash
- pip install -r backend/requirements.txt
```
4. Run the Flask App
```bash
- cd backend
- python app.py
```


5.Open the App in Your Browser
```bash
- http://localhost:5000


