# 📈 AI Stock Prediction Dashboard

A deep learning-powered stock prediction system with a performance-tracking dashboard. Uses LSTM neural networks trained on individual company data to forecast stock prices and visualize outcomes.

---

## 🧠 Objective

Predict future stock prices for top Indian companies using LSTM models and visualize performance using a dashboard interface.

---

## 🗂️ Project Structure
<pre>
AI-Stock-Prediction-Dashboard/
├── models/               # Trained LSTM model files (per company)
├── scalers/              # Saved scalers used for each stock
├── performance/          # Model performance metrics and evaluation charts
├── notebooks/            # Jupyter Notebooks for preprocessing, training, and evaluation
├── app.py                # Dashboard UI files (Flask)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
</pre>


---

## 📊 Companies Tracked

- RELIANCE.NS
- HDFCBANK.NS
- TCS.NS
- BHARTIARTL.NS
- ICICIBANK.NS
- SBIN.NS
- INFY.NS
- BAJFINANCE.NS
- HINDUNILVR.NS
- ITC.NS
- AXISBANK.NS
- HCLTECH.NS
- KOTAKBANK.NS
- M&MFIN.NS
- MARUTI.NS
- NTPC.NS
- SUNPHARMA.NS
- ULTRACEMCO.NS

---

## ⚙️ Model Evaluation Metrics (on test set)

| Ticker         | MAE   | RMSE  | R² Score |
|----------------|-------|-------|----------|
| RELIANCE.NS    | 17.96 | 22.73 | 0.9627   |
| HDFCBANK.NS    | 21.04 | 27.60 | 0.9319   |
| TCS.NS         | 53.56 | 69.04 | 0.9542   |
| BHARTIARTL.NS  | 31.36 | 40.60 | 0.8391   |
| ICICIBANK.NS   | 16.55 | 22.27 | 0.8727   |
| SBIN.NS        | 10.16 | 13.10 | 0.9141   |
| INFY.NS        | 25.92 | 32.99 | 0.9719   |
| BAJFINANCE.NS  | 105.49| 142.23| 0.9637   |
| HINDUNILVR.NS  | 30.82 | 41.29 | 0.9626   |
| ITC.NS         | 4.82  | 6.65  | 0.9617   |
| AXISBANK.NS    | 17.61 | 23.40 | 0.9206   |
| HCLTECH.NS     |       |       |          |
| KOTAKBANK.NS   |       |       |          |
| M&MFIN.NS      |       |       |          |
| MARUTI.NS      |       |       |          |
| NTPC.NS        |       |       |          |
| SUNPHARMA.NS   |       |       |          |
| ULTRACEMCO.NS  |       |       |          |

---

## 🚀 Features

- 📥 Historical stock data fetched using `yfinance`
- 🧼 Data preprocessing and normalization
- 🧠 LSTM model per company
- 📉 Evaluation using MAE, RMSE, R² Score
- 📈 Performance visualizations
- 🖥️ Interactive dashboard (Flask or Streamlit)

---

## 🧪 How to Run

### 1. Clone the Repo

```bash
git clone https://github.com/sachin-k-prajapati/AI-Stock-Prediction-Dashboard.git
cd AI-Stock-Prediction-Dashboard
```
### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate         # On Windows
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
python app.py
```

## 📁 Requirements
All packages are listed in ```requirements.txt```. Major dependencies:
```tensorflow```
```scikit-learn```
```pandas```, ```numpy```
```yfinance```, ```ta```
```matplotlib```, ```plotly```

## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.

## ⭐ Star the Repo
If this project helped you, give it a ⭐!
It helps more people discover the work.
