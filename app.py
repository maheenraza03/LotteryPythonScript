# Importing the correct packages
from flask import Flask, request, render_template
import numpy as np
import torch
from LotteryPythonScript import fetch_lotto_max_results, LottoModel, train_model, predict_next_numbers  # Adjusted imports

app = Flask(__name__)

# route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# route for the prediction page
@app.route('/predict', methods=['POST'])
# Function: predict_route
# Gets an input from the user for a year and validates it
# Then uses it to Fetch Lottery data for that year
# It also parses the results to get the 6 winning numbers
def predict_route():
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
            try:
                nums = [int(n) for n in winning_numbers]
                all_numbers.append(nums)
            except ValueError:
                continue

    if len(all_numbers) < 5:
        return render_template('index.html', error="Not enough data for training. Please try a different year.")

    # This part prepares the dataset using sliding window
    # For each group of 3 consecutive draws aka the window, the next draw is the prediction target
    max_lotto_number = 49
    window_size = 3

    X = []
    y = []
    for i in range(len(all_numbers) - window_size):
        window = np.array(all_numbers[i:i + window_size]).flatten()
        target = np.array(all_numbers[i + window_size])
        X.append(window)
        y.append(target)

    X = np.array(X) / max_lotto_number
    y = np.array(y) - 1

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Define and train the model
    model = LottoModel(input_size=window_size * 6)
    train_model(model, X_tensor, y_tensor, num_epochs=150, learning_rate=0.001)

    # Predicting the next draw
    last_window = np.array(all_numbers[-window_size:]).flatten() / max_lotto_number
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        predicted_indices = torch.argmax(output, dim=2).squeeze(0).numpy()
        predicted_numbers = (predicted_indices + 1).astype(int).tolist()

    # Returns the results on the results page
    return render_template('results.html',
                           last_numbers=all_numbers[0],
                           predicted_numbers=predicted_numbers)

if __name__ == '__main__':
    app.run(debug=True)