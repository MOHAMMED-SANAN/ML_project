import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# TITLE
# =========================

st.title("📈 Future Stock Price Prediction")

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("forcaste.csv")

data["Date"] = pd.to_datetime(data["Date"])

# =========================
# SELECT STOCK
# =========================

stock_name = st.selectbox(
    "Select Stock",
    data["Ticker"].unique()
)

# =========================
# SELECT FUTURE DATE
# =========================

future_date = st.date_input(
    "Select Future Date"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    f"Ticker_{stock_name}_xgb_model.pkl"
)

# =========================
# FILTER STOCK
# =========================

stock = data[
    data["Ticker"] == stock_name
].copy()

stock = stock.sort_values("Date")

# =========================
# LATEST DATA
# =========================

latest = stock.iloc[-1].copy()

current_price = latest["Close"]

# =========================
# UPDATE DATE FEATURES
# =========================

future_date = pd.to_datetime(future_date)

latest["Day"] = future_date.day
latest["Month"] = future_date.month
latest["weakday"] = future_date.weekday()

# =========================
# FEATURE COLUMNS
# =========================

feature_columns = [
    'Open',
    'High',
    'Low',
    'Volume',
    'price_Range',
    'price_change',
    'Return',
    'MA_7',
    'MA_30',
    'EMA_7',
    'EMA_30',
    'Rolling_STD',
    'Lag_1',
    'Lag_2',
    'Lag_3',
    'Lag_4',
    'Lag_5',
    'Lag_6',
    'Lag_7',
    'Day',
    'Month',
    'weakday',
    'Return_Lag1',
    'Return_Lag5',
    'MA_5_Ratio',
    'MA_20_Ratio'
]

# =========================
# HANDLE NULL VALUES
# =========================

latest = latest.fillna(0)

# =========================
# CREATE FEATURES
# =========================

features = np.array([
    [latest[col] for col in feature_columns]
])

# =========================
# PREDICT
# =========================

if st.button("Predict Future Price"):

    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")

    st.write(f"Current Price: ₹{current_price:.2f}")

    st.write(f"Predicted Price: ₹{prediction:.2f}")

    # =========================
    # PROFIT OR LOSS
    # =========================

    difference = prediction - current_price

    percentage = (difference / current_price) * 100

    if prediction > current_price:

        st.success(
            f"📈 Expected Profit: ₹{difference:.2f} ({percentage:.2f}%)"
        )

    elif prediction < current_price:

        st.error(
            f"📉 Expected Loss: ₹{abs(difference):.2f} ({abs(percentage):.2f}%)"
        )

    else:

        st.info("No Price Change Expected")
        
    chart_data = pd.DataFrame({
        "Price": [current_price, prediction]
    },
    index=["Current Price", "Predicted Price"])

    st.line_chart(chart_data)