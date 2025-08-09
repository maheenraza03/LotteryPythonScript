# importing the correct packages 
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# api key from rapidapi from "Canda Lottery"
API_KEY = "ur api key"

# defining the headers for the api link
headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "canada-lottery.p.rapidapi.com"
}

# Function: fetch_lotto_max_results
# This function is to grab the winning numbers of a specific year from the RapidAPI url
# The year is inputted by the user
def fetch_lotto_max_results(year):
    url = f"https://canada-lottery.p.rapidapi.com/6-49/years/{year}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


# Class: LottoModel
# This class is a custom neural network that subclasses PyTorch
# Defines a feedforward neural network
# The input is historical features from previous lottery draws
class LottoModel(nn.Module):
    def __init__(self, input_size=18):
        super(LottoModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6 * 49)

    # Forward Function
    # Used to predict probablity distributions for 49 different possible numbers for each of the 6 drawns numbers for Lotto 6/49
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 6, 49)
        return x

# Function: train_model
# Train the neural network
def train_model(model, X_tensor, y_tensor, num_epochs=150, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)

        loss = 0
        for i in range(6):
            loss += criterion(outputs[:, i, :], y_tensor[:, i])

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Function: predict_next_numbers
# Using the trained PyTorch model, it predicts the next set of lottery numbers based on some input features 
# Predicts 6 different numbers from 1 to 49
def predict_next_numbers(model, previous_numbers):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(previous_numbers, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        predicted_indices = torch.argmax(output, dim=2).squeeze(0).numpy()
        predicted_numbers = predicted_indices + 1 
        return predicted_numbers

# Main Function
# Asks the user to input the year of their choice
# It then goes through the above functions 
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    user_year = input("Enter a year to fetch 649 results (e.g., 2025): ")

    try:
        user_year = int(user_year)
        lotto_results = fetch_lotto_max_results(user_year)

        if lotto_results:
            valid_draws = []

            for draw in lotto_results:
                date_str = draw.get("date")
                numbers = draw.get("classic", {}).get("numbers", [])
                if len(numbers) == 6 and date_str:
                    try:
                        draw_date = datetime.strptime(date_str, "%Y-%m-%d")
                        valid_draws.append((draw_date, [int(n) for n in numbers]))
                    except ValueError as e:
                        print(f"Skipping due to date error: {e}")
                        continue

            if len(valid_draws) < 5:
                print("Not enough valid draws to train the model.")
                return
            
            # i wanted to see the most recent draws to see what it would return
            print("\n--- Most Recent Draws ---")
            for date, nums in valid_draws[:5]:
                print(f"{date.date()}: {nums}")
            print("-------------------------\n")

            window_size = 3
            max_lotto_number = 49

            draws_only = [nums for _, nums in valid_draws]

            X = []
            y = []

            for i in range(len(draws_only) - window_size):
                window = np.array(draws_only[i:i + window_size]).flatten()
                target = np.array(draws_only[i + window_size])
                X.append(window)
                y.append(target)

            X = np.array(X) / max_lotto_number 
            y = np.array(y) - 1 

            data_tensor = torch.tensor(X, dtype=torch.float32)
            labels_tensor = torch.tensor(y, dtype=torch.long)

            model = LottoModel(input_size=window_size * 6)
            train_model(model, data_tensor, labels_tensor, num_epochs=100, learning_rate=0.001)

            last_window = np.array(draws_only[-window_size:]).flatten() / max_lotto_number
            predicted_numbers = predict_next_numbers(model, last_window)

            print(f"Last winning numbers on {valid_draws[0][0].date()}: {draws_only[0]}")
            print(f"Predicted next winning numbers: {predicted_numbers}\n")
        else:
            print("No data fetched.")
    except ValueError:
        print("Please enter a valid year.")

if __name__ == "__main__":
    main()
