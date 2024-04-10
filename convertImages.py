import cv2
import numpy as np
from scipy import io

def blur_image(image, A, B):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create blur kernel matrices
    # A_r * X * A_c.T
    # Where X is the desired image(Original images) + eta (Noise)
    # Keep images 256^2 = 65536 pixels
    G = np.dot(np.dot(B, gray_image), A.T) + 0.001 * np.random.rand(256, 256)

    return G, A, B, gray_image

# Read image
# image = cv2.imread('./data/pumpkins.jpg')
# image = cv2.imread('./data/butterflies.jpg')
# image = cv2.imread('./data/origimage.jpg')
image = cv2.imread('./data/Mona_Lisa.jpeg')
mat_data = io.loadmat('./data/proj1data.mat')
A = mat_data['A']
B = mat_data['B']
# Check if image is loaded successfully
if image is None:
    print("Error: Image not loaded")
else:
    # cutting images 256x256 if is too big
    # I'm Cutting the image to 256x256, but from the top left corner
    # If you want to cut from the center, you can change the values of the slice
    # Or if you cut from  any other corner, you can change the values of the slice
    # Just in case you wanna try.
    cropped_image = image[:256, :256]
    # Apply blur effect
    g, a, b, gray = blur_image(cropped_image, A, B)

    io.savemat('data/matrices.mat', {'G': g, 'A': A, 'B': B})

    # Save images to files
    cv2.imwrite('blurred_image.jpg', g)
    cv2.imwrite('original_image.jpg', cropped_image)
    cv2.imwrite('gray_image.jpg', gray)
    print("Images saved successfully.")

