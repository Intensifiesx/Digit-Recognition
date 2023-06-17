# %%
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import os

# %%
folder_path = "ImageDataset"  # Specify the path to your folder here
files = os.listdir(folder_path)
print(files)


# %%
def print_image():
    # Print the resulting array
    for row in array:
        for cell in row:
            if cell == 0:
                print(cell, end=" ")
            else:
                print("\033[91m" + str(cell) + "\033[0m", end=" ")
        print()


df = pd.DataFrame()
# %%
for file in files:
    # Process each image file
    file_path = os.path.join(folder_path, file)
    print("Processing:", file_path)
    # Add your code to process the image file here

    # Load the image and convert it to grayscale
    image = Image.open(file_path).convert("L").resize((16, 16))

    # Convert each pixel to a binary value (1 or 0)
    pixels = []
    for y in range(16):
        for x in range(16):
            pixel = image.getpixel((x, y))
            binary = 1 if pixel < 128 else 0
            pixels.append(binary)

    # Reshape the list into a 16x16 array
    array = [pixels[i : i + 16] for i in range(0, 256, 16)]  # Python list
    print_image()

    # Make it NUMPY array
    array = np.array(array).ravel().reshape(1, -1)
    df = df.append(pd.DataFrame(array))

    # %%
