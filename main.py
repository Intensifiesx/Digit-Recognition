#
# Title: Number Recognition
# Description: A program that uses a random forest classifier to recognize numbers
# Authors: Zain Hindi, Alex Akoopie
# Date: 2023-06-18
#

# %%
# ======Libraries======
import os
import threading
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# %%
# ======Function Definitions======
def get_image(path):  # Get Image Function
    image = Image.open(path)  # Loads the image
    image = image.convert("RGBA")  # Converts the image to RGBA
    image = image.resize((28, 28))  # Resizes the image to 28x28 (if necessary)
    pixels = []  # List representing every 4x4 pixel in a 28x28 image
    # Loops through every 2 pixels for each image
    for y in range(0, 28, 2):
        for x in range(0, 28, 2):
            pixels.append(0)  # Appends a new chunk of 4x4 pixels
            # Loops through each pixel for each 4x4 chunk
            for i in range(2):
                for j in range(2):
                    pixel = image.getpixel((x + j, y + i))
                    binary = 0 if pixel[3] == 0 else 1
                    # Adds the number of pixels that are active in each chunk (0-4)
                    pixels[-1] += binary
    # Turns the pixels list into columns
    array = [pixels[i : i + 14] for i in range(0, 196, 14)]
    # Transforms the Python list into a NumPy array
    return np.array(array).ravel().reshape(1, -1)


# %%
# ======Data Loading======
df = pd.DataFrame()
label = pd.DataFrame()
directory = "ImageDataset"
# Loops through the ImageDataset directory
for index, folder in enumerate(os.listdir(directory)):
    subdirectory = os.path.join(directory, folder)
    # Loops through each folder in the ImageDataset directory
    for subindex, file in enumerate(os.listdir(subdirectory)):
        # Prematurely ends the loop to make loading faster
        if subindex >= 1000:
            break
        file_path = os.path.join(subdirectory, file)
        array = get_image(file_path)  # Gets the image as an array of pixels
        # Adds the array to the dataframe as a new row
        df = df.append(pd.DataFrame(array))
        label = label.append(pd.DataFrame([index]))  # Keeps track of the label


# %%
# ======Initialization======
df.insert(0, "Number", label)  # Appends the label to the first column of the dataframe
features = df.drop("Number", axis=1)  # Gets the features
x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=42
)  # Splits the data into training and testing sets


# %%
# ======Testing======
model = RandomForestClassifier(random_state=42)  # Defines the model
model.fit(x_train, y_train)  # Fits the parameters to the model
predictions = model.predict(x_test)  # Gets accuracy score from test set
print(f"Model Accuracy:\t{accuracy_score(y_test, predictions)}")

# ======Training======
for file in os.listdir("Test"):
    print("=" * 20)
    print(f"Testing {file}...")

    # PATH TO TESTING IMAGE
    test_path = "Test/" + file

    # Gets testing image
    test = pd.DataFrame()
    array = get_image(test_path)
    test = test.append(pd.DataFrame(array))

    # Gets and prints result
    predicted_probabilities = model.predict_proba(test)[0]
    predicted_number = model.predict(test)[0]
    print("Chance of it being a...")
    for i in range(1, 11):
        print(
            f"\033[92m{i-1}: {predicted_probabilities[i-1] * 100:05.2f}%",
            end="\t" if i % 3 else "\n",
        )
    print(f"\033[37mHmmmm... I think the number is a {predicted_number}")
# %%
