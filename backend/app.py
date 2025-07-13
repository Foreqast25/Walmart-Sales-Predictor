from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import joblib
import random
import pandas as pd
from flask import request, jsonify
from datetime import datetime, timedelta
from calendar import monthrange

# ============================
# Load actual sales data
# ============================
actual_sales_df = pd.read_csv("actual_sales.csv")

# ============================
# Flask Setup
# ============================
app = Flask(__name__)
app.secret_key = "supersecretkey"
CORS(app)

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Dummy user store
users = {
    "manager@store.com": {
        "password": "1234",
        "store": "Hyderabad",
        "name": "Manager Rao"
    },
    "arpit@store.com": {
        "password": "5678",
        "store": "Jammu",
        "name": "Manager Arpit"
    }
}

# ============================
# Helper: Fetch Actual Sales
# ============================
def get_actual_sales(input_dict):
    match = actual_sales_df[
        (actual_sales_df['Store'] == input_dict['Store']) &
        (actual_sales_df['Item'] == input_dict['Item']) &
        (actual_sales_df['Month'] == input_dict['Month']) &
        (actual_sales_df['Week'] == input_dict['Week'])
    ]
    return match.iloc[0]['Sales'] if not match.empty else None

def build_features_for_date(store, item, date_obj):
    features = {}

    # Base inputs
    features['Store'] = store
    features['Item'] = item
    features['Price'] = 85     # 🔁 Replace or randomize if needed
    features['Discount'] = 10
    features['Temperature'] = 32
    features['Rainfall'] = 10
    features['IsHoliday'] = 0
    features['Is_Ugadi'] = 0
    features['Is_Winter'] = 0

    # Date-based
    features['Month'] = date_obj.month
    features['Week'] = date_obj.isocalendar()[1]
    features['Day'] = date_obj.day
    features['Quarter'] = (date_obj.month - 1) // 3 + 1
    features['DayOfWeek'] = date_obj.weekday()

    # Encodings
    item_map = {
        "Cold Beverages": 0, "Dry Fruits": 1, "Mosquito Repellents": 2, "Packed Bottled Water": 3,
        "Rice": 4, "Woolen Jackets": 5, "Pulses": 6, "Spices": 7, "Edible Oils": 8, "Bath Towels": 9,
        "Toilet Cleaners": 10, "Cornflakes": 11, "Papad": 12, "Sugar": 13, "Tea Powder": 14, "Cookies": 15
    }
    features['Item_Encoded'] = item_map.get(item, 0)

    features['Store_Encoded'] = 0 if store == "Hyderabad" else 1

    # 🔁 Add missing fields with default values
    features['StoreType_Encoded'] = 0  # 0: Metro, 1: Tourism (default to Metro)
    features['Region_Encoded'] = 1     # South by default; modify as needed

    # Derived binary flags
    features['Discount_Flag'] = 1 if features['Discount'] > 0 else 0
    features['Season_Summer'] = 1 - features['Is_Winter']
    features['Season_Winter'] = features['Is_Winter']
    features['Rain_Bin_Low'] = 1 if features['Rainfall'] < 30 else 0
    features['Rain_Bin_Medium'] = 1 if 30 <= features['Rainfall'] <= 60 else 0
    features['Temp_Bin_Hot'] = 1 if features['Temperature'] >= 35 else 0
    features['Temp_Bin_Mild'] = 1 if 20 <= features['Temperature'] < 35 else 0

    return features


# ============================
# Routes
# ============================
@app.route('/')
def home():
    name = session.get('name', 'Manager')
    return render_template("index.html", user_name=name)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email]['password'] == password:
            session['user'] = email
            session['name'] = users[email]['name']
            return redirect(url_for('dashboard'))
        else:
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        store = request.form['store']
        if email in users:
            return render_template("signup.html", error="User already exists")
        users[email] = {"password": password, "store": store, "name": store}
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    print(session)  # 👈 See if 'name' is in there
    return render_template("dashboard.html", user_name=session.get('name', 'Manager'))

@app.route('/eco_stock', methods=['GET', 'POST'])
def eco_stock():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("eco_stock.html", user_name=session.get('name', 'Manager'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_dict = data['features']

    column_order = [
        'Month', 'Week', 'IsHoliday', 'Price', 'Temperature', 'Rainfall', 'Is_Ugadi', 'Is_Winter',
        'Discount', 'Store_Encoded', 'Item_Encoded', 'StoreType_Encoded', 'Region_Encoded',
        'Day', 'Quarter', 'DayOfWeek', 'Discount_Flag', 'Season_Summer', 'Season_Winter',
        'Rain_Bin_Low', 'Rain_Bin_Medium', 'Temp_Bin_Hot', 'Temp_Bin_Mild'
    ]

    missing = [col for col in column_order if col not in input_dict]
    if missing:
        return jsonify({'error': f'Missing input fields: {missing}'}), 400

    input_df = pd.DataFrame([[input_dict[col] for col in column_order]], columns=column_order)
    prediction = model.predict(input_df)[0]
    actual_sales = get_actual_sales(input_dict)

    return jsonify({
        'prediction': round(float(prediction), 2),
        'actual': round(float(actual_sales), 2) if actual_sales is not None else None
    })

@app.route('/sales_trend', methods=['GET'])
def sales_trend():
    store = request.args.get('store')
    item = request.args.get('item')
    if not store or not item:
        return jsonify({'error': 'Missing "store" or "item" parameter'}), 400

    filtered_df = actual_sales_df[
        (actual_sales_df['Store'] == store) &
        (actual_sales_df['Item'] == item)
    ]

    if filtered_df.empty:
        return jsonify({'months': [], 'actual_sales': []})

    trend_df = filtered_df.groupby('Month')['Sales'].mean().reset_index().sort_values(by='Month')

    return jsonify({
        'months': trend_df['Month'].tolist(),
        'actual_sales': trend_df['Sales'].round(2).tolist()
    })

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================
# Dummy Chatbot Assistant
# ============================
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get("message", "").lower()

    dummy_responses = {
        "what are the sales for past month": "📊 Sales for the past month averaged 320 units across all stores.",
        "forecast for next week": "🔮 Based on our model, sales are expected to rise by 12% next week due to regional discounts and seasonal factors.",
        "help with eco stock": "♻️ If your stock is above 500 units, consider rerouting or applying eco-discounts. For stock under 100, no action is needed.",
        "reset chatbot": "✅ Chatbot memory reset. Ask me anything!",
        "who made you": "🤖 I was handcrafted with love by Team Walmart Forecasting AI ❤️"
    }

    for key, reply in dummy_responses.items():
        if key in user_message:
            return jsonify({"response": reply})

    return jsonify({"response": "🤔 Sorry, I can only answer predefined questions for now. Try asking about past sales, forecasts, or eco stock help."})

@app.route('/calendar')
def calendar():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("calendar.html", user_name=session.get('name', 'Manager'))

@app.route('/calendar_events')
def calendar_events():
    # Dummy data
    events = [
        {"title": "Predicted: 320 units", "start": "2025-07-15"},
        {"title": "Predicted: 295 units", "start": "2025-07-16"},
    ]
    return jsonify(events)


@app.route("/get_calendar_events")
def get_calendar_events():
    store = request.args.get("store")
    item = request.args.get("item")

    if not store or not item:
        return jsonify([]), 400

    today = datetime.today()
    year = today.year
    month = today.month

    # Get the number of days in the current month
    _, last_day = monthrange(year, month)
    end_date = datetime(year, month, last_day)

    events = []
    current_day = today

    while current_day <= end_date:
        sales = random.randint(40, 75)
        events.append({
            "title": f"{item}\nForecast: {sales}",
            "start": current_day.strftime('%Y-%m-%d')
        })
        current_day += timedelta(days=1)

    return jsonify(events)

# ============================
# Run App
# ============================
if __name__ == "__main__":
    app.run(debug=True)
