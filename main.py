#
# Title: Number Recognition
# Description: A program that uses a random forest classifier to recognize numbers
# Authors: Zain Hindi, Alex Akoopie
# Date: 2023-06-18
#

# %%
# ======Libraries======
import os
import multiprocessing
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

NUM_PROCESSES_DIRECTORIES = 2  # Number of processes for the directories
IMAGES_TO_PROCESS = 1000  # Number of images to process

startTime = time.time()  # Gets the start time of the program


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


def directory(index, shared_df, shared_label, lock):
    print(f"Process {index} started")
    for i in range(index, 10, NUM_PROCESSES_DIRECTORIES):
        subdirectory = "ImageDataset" + "/" + str(i)
        for j in range(IMAGES_TO_PROCESS):
            array = getImage(subdirectory + "/" + str(j) + ".png")
            lock.acquire()
            shared_df.append(pd.DataFrame(array))
            shared_label.append(pd.DataFrame([i]))
            lock.release()
    print(f"Process {index} finished")


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_df = manager.list()
    shared_label = manager.list()
    lock = manager.RLock()

    processes = [
        multiprocessing.Process(
            target=directory, args=(i, shared_df, shared_label, lock)
        )
        for i in range(NUM_PROCESSES_DIRECTORIES)
    ]

    start_time = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    end_time = time.time()

    df = pd.concat(shared_df)
    label = pd.concat(shared_label)

    print(
        f"Time to load n={IMAGES_TO_PROCESS} data ({NUM_PROCESSES_DIRECTORIES} Multi-Processes): {end_time - start_time:05.2f}s"
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
    array = getImage(test_path)
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
