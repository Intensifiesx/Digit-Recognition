# %%
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import os
import random
import warnings

# for row in array:
#     for cell in row:
#         if cell == 0:
#             print(cell, end=" ")
#         else:
#             print(
#                 f"{random.choice(my_colors)}" + str(cell) + "\033[0m", end=" "
#             )
#     print()

warnings.filterwarnings("ignore")

my_colors = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
df = pd.DataFrame()
label = df

# print the files in each folder
for i in range(os.listdir("ImageDataset/").__len__()):
    files = os.listdir("ImageDataset/" + str(i))
    # Process each image file
    for file in files[:500]:
        file_path = "ImageDataset/" + str(i) + "/" + file
        # Load the image and convert it to grayscale
        image = Image.open(file_path).convert("RGBA").resize((28, 28))
        # Convert each pixel to a binary value (1 or 0)
        pixels = []
        for y in range(28):
            for x in range(28):
                pixel = image.getpixel((x, y))
                binary = 0 if pixel[3] == 0 else 1
                pixels.append(binary)
        array = [pixels[i : i + 28] for i in range(0, 784, 28)]  # Python list
        # Make it NUMPY array
        array = np.array(array).ravel().reshape(1, -1)
        df = df.append(pd.DataFrame(array))
        label = label.append(pd.DataFrame([i]))

# %%
df.insert(0, "Number", label)
# %%
df = sklearn.utils.shuffle(df)  # Shuffles the data frame

features = df.drop("Number", axis=1)
# %%
# Setup arrays to store training and test accuracies
neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    knn.fit(features, label)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(features, label)

# %%
model = KNeighborsClassifier(n_neighbors=4021)
model = model.fit(features, label)
# We already have our label
# %%
# ====TESTING====
ghetto = "Test/Image2.png"
test = pd.DataFrame()
image = Image.open(file_path).convert("RGBA").resize((28, 28))
# Convert each pixel to a binary value (1 or 0)
pixels = []
for y in range(28):
    for x in range(28):
        pixel = image.getpixel((x, y))
        binary = 0 if pixel[3] == 0 else 1
        pixels.append(binary)
array = [pixels[i : i + 28] for i in range(0, 784, 28)]  # Python list
array = np.array(array).ravel().reshape(1, -1)
test = test.append(pd.DataFrame(array))
predicted_probabilities = model.predict_proba(test)
print(f"Hmmmm... I think the number is a {model.predict(test)[0]}?")
# %%
