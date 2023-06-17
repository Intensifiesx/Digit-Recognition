from PIL import Image

# Load the image
image = Image.open("image.png")

# Convert the image to grayscale
image = image.convert("L")

# Resize the image to 16x16 if necessary
image = image.resize((16, 16))

# Convert each pixel to a binary value (1 or 0)
pixels = []
for y in range(16):
    for x in range(16):
        pixel = image.getpixel((x, y))
        binary = 1 if pixel < 128 else 0
        pixels.append(binary)

# Reshape the list into a 16x16 array
array = [pixels[i : i + 16] for i in range(0, 256, 16)]

# Print the resulting array
for row in array:
    print(" ".join(map(str, row)))
