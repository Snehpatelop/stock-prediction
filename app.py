
from flask import Flask, render_template, request, redirect, url_for
from model import predict_stock_price
import datetime

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Get today's date for the max date in the form
    today = datetime.date.today()
    max_date = today.strftime('%Y-%m-%d')
    
    # Default start date (5 years ago)
    default_start = (today - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    return render_template('index.html', max_date=max_date, default_start=default_start)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        stock_ticker = request.form['stock_ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        try:
            result = predict_stock_price(stock_ticker, start_date, end_date)
            
            if result is None:
                error_message = "Prediction failed. Please check the stock ticker or dates."
                return render_template('index.html', error=error_message)
            
            price, actual_price, performance_metrics, plot_path = result
            
            return render_template('result.html', 
            stock_ticker=stock_ticker,
            predicted_price=price,
            actual_price=actual_price,
            rmse=performance_metrics['rmse'],
            r2=performance_metrics['r2'],
            mae=performance_metrics['mae'],
            mape=performance_metrics['mape'],
            plot_filename=plot_path  # <<< USE plot_filename here correctly
)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render_template('index.html', error=error_message)

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
