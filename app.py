from flask import Flask, request, render_template, jsonify
import numpy as np
import torch
from LotteryPythonScript import fetch_lotto_max_results, LottoPredictionModel, train_model, predict_next_numbers

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = request.form.get('year')
    if not year or not year.isdigit():
        return render_template('index.html', error="Invalid year input. Please enter a valid year.")

    year = int(year)
    lotto_results = fetch_lotto_max_results(year)
    if not lotto_results:
        return render_template('index.html', error="Failed to fetch lottery results. Please try again later.")

    all_numbers = []
    for draw in lotto_results:
        winning_numbers = draw.get("classic", {}).get("numbers", [])
        if len(winning_numbers) == 6:
            all_numbers.append(winning_numbers)

    if len(all_numbers) < 2:
        return render_template('index.html', error="Not enough data for training. Please try a different year.")

    # Normalize data for training
    data = np.array(all_numbers[:-1])
    labels = np.array(all_numbers[1:])
    data = data / np.max(data)
    labels = labels / np.max(labels)

    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Train the model
    model = LottoPredictionModel()
    train_model(model, data_tensor, labels_tensor, num_epochs=100, learning_rate=0.001)

    # Predict the next numbers
    last_numbers = all_numbers[-1]
    predicted_numbers = predict_next_numbers(model, last_numbers)
    predicted_numbers = (predicted_numbers * np.max(data)).astype(int).tolist()

    # Render the results page
    return render_template('results.html', last_numbers=last_numbers, predicted_numbers=predicted_numbers)

if __name__ == '__main__':
    app.run(debug=True)
