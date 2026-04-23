from flask import Flask, render_template, request, redirect, session
import os
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)
app.secret_key = "secret123"
DB_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.getcwd()), "UPI Fraud Detection")
DB_PATH = os.path.join(DB_DIR, "database.db")


def get_db_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=10)

# -------- LOAD MODEL --------
model = pickle.load(open('models/random_forest_model.pkl', 'rb'))

# -------- INIT DB --------
def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # users table
    c.execute('''CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )''')

    # transactions table
    c.execute('''CREATE TABLE IF NOT EXISTS transactions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL,
        oldbalance REAL,
        newbalance REAL,
        result TEXT,
        risk REAL
    )''')

    conn.commit()
    conn.close()

init_db()

# -------- LOGIN --------
@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('user')
        password = request.form.get('pass')

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, password))
        data = c.fetchone()
        conn.close()

        if data:
            session['user'] = user
            return redirect('/home')

    return render_template('login.html')

# -------- SIGNUP --------
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        user = request.form.get('user')
        password = request.form.get('pass')

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO users(username, password) VALUES(?,?)", (user, password))
        conn.commit()
        conn.close()

        return redirect('/')

    return render_template('signup.html')

# -------- HOME --------
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html')

# -------- PREDICT --------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form.get('amount', 0))
        old = float(request.form.get('oldbalanceOrg', 0))
        new = float(request.form.get('newbalanceOrig', 0))
    except ValueError:
        return render_template(
            'result.html',
            result="Invalid input",
            risk=0,
            reason="Please enter valid numeric values."
        )

    x = np.array([[amount, old, new]])

    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0][1]
    risk = round(prob * 100, 2)

    result = "Fraud Alert" if pred == 1 else "Safe"

    reason = ""
    if amount > 50000:
        reason += "High Amount, "
    if new < old:
        reason += "Balance mismatch"

    # Save to DB. If the database is locked/read-only, still show the result page.
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO transactions(amount, oldbalance, newbalance, result, risk) VALUES(?,?,?,?,?)",
                  (amount, old, new, result, risk))
        conn.commit()
        conn.close()
    except sqlite3.Error as error:
        reason = f"{reason} Database save skipped: {error}".strip()

    return render_template('result.html', result=result, risk=risk, reason=reason)

# -------- DASHBOARD --------
@app.route('/dashboard')
def dashboard():
    try:
        conn = get_db_connection()
        data = conn.execute("SELECT * FROM transactions").fetchall()
        conn.close()
    except sqlite3.Error:
        data = []

    fraud = sum(1 for i in data if "Fraud" in str(i[4]))
    safe = len(data) - fraud

    return render_template('dashboard.html', data=data, fraud=fraud, safe=safe)

# -------- LOGOUT --------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

# -------- RUN --------
if __name__ == "__main__":
    app.run(debug=True)
