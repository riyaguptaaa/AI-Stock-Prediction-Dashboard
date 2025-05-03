from flask import Flask, render_template, request, jsonify
from predictor import predict_price, get_stock_analysis, calculate_risk_assessment, calculate_confidence_score
import yfinance as yf
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Available tickers from your models directory
TICKERS = [f.replace('.h5', '') for f in os.listdir('models') if f.endswith('.h5')]

@app.route('/')
def home():
    return render_template('index.html', tickers=TICKERS)

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    try:
        # Get 3 months of data for prediction
        df = yf.download(ticker, period='1y')
        
        if len(df) < 60:
            raise ValueError("Not enough historical data (need ≥60 days)")
        
        # Get prediction and analysis
        prediction = predict_price(ticker, df)
        analysis = get_stock_analysis(ticker, df, prediction['accuracy_calc'])
        
        analysis['confidence_score'] = calculate_confidence_score(analysis['reasons'], analysis)
        analysis['risk_score'] = calculate_risk_assessment(analysis)
        analysis['accuracy'] = prediction['accuracy_calc']
        
        prediction = prediction['predicted']
        
        
        return render_template('results.html',
                            ticker=ticker,
                            prediction=prediction,
                            analysis=analysis)
    
    except Exception as e:
        return render_template('index.html',
                            tickers=TICKERS,
                            error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)