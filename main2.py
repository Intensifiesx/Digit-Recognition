#
# Title: Number Recognition
# Description: A program that uses a random forest classifier to recognize numbers
# Authors: Zain Hindi, Alex Akoopie
# Date: 2023-06-18
#

# %%
# ======Libraries======
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

startTime = time.time()  # Gets the start time of the program
IMAGES_TO_PROCESS = 2000  # Number of images to process


# %%
# ======Data Loading======
def getImage(path):
    image = Image.open(path).convert("RGBA").resize((28, 28))
    pixels = []
    for y in range(0, 28, 2):
        for x in range(0, 28, 2):
            pixels.append(0)
            for k in range(2):
                for o in range(2):
                    pixel = image.getpixel((x + k, y + o))
                    binary = 0 if pixel[3] == 0 else 1
                    pixels[-1] += binary
    # Turns the pixels list into columns
    array = [pixels[i : i + 14] for i in range(0, 196, 14)]
    # Transforms the Python list into a NumPy array
    return np.array(array).ravel().reshape(1, -1)


df = pd.DataFrame()
label = pd.DataFrame()
directory = "ImageDataset"
# Loops through the ImageDataset directory
for i in range(10):
    subdirectory = "ImageDataset" + "/" + str(i)
    for j in range(IMAGES_TO_PROCESS):
        array = getImage(subdirectory + "/" + str(j) + ".png")
        df = pd.concat([df, pd.DataFrame(array)], ignore_index=True)
        label = pd.concat([label, pd.DataFrame([i])], ignore_index=True)

endTime = time.time()  # Gets the end time of the program
print(
    f"Time to load n={IMAGES_TO_PROCESS} data (No Multi-Processing): {endTime - startTime:05.2f}s"
)
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
