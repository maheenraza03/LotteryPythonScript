import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# api key for accessing the lottery api
API_KEY = "YOUR_API_KEY_HERE"

# headers (based on the rapidapi code snippets)
headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "canada-lottery.p.rapidapi.com"
}

# get the lottery results from the lottery api and get the json response file
def fetch_lotto_max_results(year):
    # api URL
    url = f"https://canada-lottery.p.rapidapi.com/6-49/years/{year}"
    # get the response from the api
    response = requests.get(url, headers=headers)

    # if the response is successful, retrieve the JSON with the response
    if response.status_code == 200:
        return response.json()  
    # otherwise print an error message
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# defining the neural network
class LottoPredictionModel(nn.Module):
    def __init__(self):
        super(LottoPredictionModel, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # the input layer (takes in the 6 previous winning numbers)
        self.fc2 = nn.Linear(128, 64) # the hidden layer (analyzes the complex relationships between the inputs, transforms the data into a smaller feature space)
        self.fc3 = nn.Linear(64, 6)   # the output layer (makes an output for the 6 predicted winning numbers based on the 64 features from the hidden layer)
        self.dropout = nn.Dropout(0.5) # helps prevent overfitting

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # pass the input data through the input layer, and then apply the activation function
        x = self.dropout(x) # drop 50% of the features
        x = torch.relu(self.fc2(x))  # take the output of the input layer and pass it through the hidden layer, and then apply the activation function
        x = self.fc3(x)  # finally, take the output of the hidden layer and pass it through the output layer to get our new 6 predicted lottery numbers
        return x

# used to train the model
def train_model(model, data, labels, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()  # MSE loss function (calculates how close the model's predictions are compared to the actual labels)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # update the model's parameters during training 

    for epoch in range(num_epochs):
        model.train() # puts the model on training mode
        optimizer.zero_grad() # gradients from the previous training step are cleared
        outputs = model(data) # pass the data through the model
        loss = criterion(outputs, labels) # calculate the difference between predictions and labels using the MSE function
        loss.backward() # do backpropagation: computes the gradients of the loss with respect to each model parameter
        optimizer.step() # update the model parameters

# used to predict the next winning numers
def predict_next_numbers(model, previous_numbers):
    model.eval() # put the model into evaluation mode
    with torch.no_grad(): # don't compute gradients
        input_tensor = torch.tensor(previous_numbers, dtype=torch.float32).unsqueeze(0) # convert the previous numbers to tensor
        predicted_numbers = model(input_tensor).squeeze(0).numpy() # make the prediction
        
        # round the predicted numbers to whole numbers
        predicted_numbers = np.round(predicted_numbers)
        return predicted_numbers

# main function
def main():
    # ask for user input for the year
    user_year = input("Enter a year to fetch 649 results: ")

    try:
        # convert the string into an int
        user_year = int(user_year)
        # input the year into the api URL
        lotto_results = fetch_lotto_max_results(user_year)

        if lotto_results:
            # array to store winning numbers for the API
            all_numbers = []

            for draw in lotto_results:
                # extract the winning numbers from the response
                date = draw.get("date")
                winning_numbers = draw.get("classic", {}).get("numbers", [])
                # make sure it's only 6 winning numbers
                if len(winning_numbers) == 6:  
                    all_numbers.append(winning_numbers)

            # convert all numbers to numpy array and prepare for training
            data = np.array(all_numbers[:-1])  # use all except the last draw for training
            labels = np.array(all_numbers[1:])  # use the next draw as the label

            # normalize data (e.g., scale the numbers between 0 and 1)
            data = data / np.max(data)
            labels = labels / np.max(labels)

            # convert to PyTorch tensors
            data_tensor = torch.tensor(data, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # initialize and train the model
            model = LottoPredictionModel()
            train_model(model, data_tensor, labels_tensor, num_epochs=100, learning_rate=0.001)

            # get the last set of numbers to predict the next ones
            last_numbers = all_numbers[-1]  # use the most recent winning numbers
            print(f"Last winning numbers: {last_numbers}")

            # predict the next winning numbers
            predicted_numbers = predict_next_numbers(model, last_numbers)
            predicted_numbers = predicted_numbers * np.max(data)  # denormalize back to original range

            print(f"Predicted next winning numbers: {predicted_numbers.astype(int)}")  # ensure integers are displayed
        else:
            print("No data fetched.")
    except ValueError:
        print("Please enter a valid year.")

if __name__ == "__main__":
    main()
