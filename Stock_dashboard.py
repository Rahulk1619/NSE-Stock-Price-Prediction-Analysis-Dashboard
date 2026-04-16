import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Prediction Dashboard")

# ── Stock selector ─────────────────────────────────────────────
stock = st.selectbox("Select a stock", ["RELIANCE", "TCS", "INFY"])

@st.cache_data
def load_data(stock):
    conn = sqlite3.connect("data/stocks.db")
    df = pd.read_sql(f"SELECT * FROM {stock}", conn)
    conn.close()
    return df

@st.cache_resource
def train_model(stock):
    df = load_data(stock)
    features = ["Close", "MA7", "MA20", "MA50", "RSI14",
                "MACD_hist", "BB_width", "Close_lag1",
                "Close_lag2", "Close_lag5", "daily_return"]
    X = df[features].dropna()
    y = df.loc[X.index, "Target"]
    mask = y.notna()
    X, y = X[mask], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, preds

df = load_data(stock)
model, X_train, X_test, y_train, y_test, preds = train_model(stock)

# ── Metrics ────────────────────────────────────────────────────
rmse    = np.sqrt(mean_squared_error(y_test, preds))
mae     = mean_absolute_error(y_test, preds)
err_pct = (rmse / y_test.mean()) * 100

st.caption(f"Showing results for: {stock}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE",      f"₹{rmse:.2f}")
col2.metric("MAE",       f"₹{mae:.2f}")
col3.metric("Error %",   f"{err_pct:.2f}%")
col4.metric("Test rows", f"{len(y_test)}")

# ── Predicted vs Actual ────────────────────────────────────────
st.subheader("Predicted vs Actual Price")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(y_test.values, color="#2563eb", label="Actual",    linewidth=1.5)
ax1.plot(preds,         color="#f97316", label="Predicted", linewidth=1.5, linestyle="--")
ax1.set_ylabel("Price (INR)")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ── Feature importance ─────────────────────────────────────────
st.subheader("Feature Importance")
importance = pd.Series(model.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=True)
fig2, ax2 = plt.subplots(figsize=(8, 5))
importance.plot(kind="barh", ax=ax2, color="#2563eb")
ax2.set_xlabel("Importance score")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# ── Raw data ───────────────────────────────────────────────────
st.subheader("Recent Data")
st.dataframe(df[["Close", "RSI14", "MACD_hist", "BB_width", "Target"]].tail(50))