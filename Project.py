import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

file_path = "./data/proj1data" 

# Reading file
with open(file_path, 'r') as file:
    data = file.read().split()

# Function to parse the data into matrices, cutting each matrix into 256x256
# Since data is all stored in a single file, but this file does not have 256 columns,
# we need need create the matrices based on it.
def parse_matrix(data):
    matrix = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            matrix[i][j] = float(data[i * 256 + j])
    return matrix

# Split the data into matrices A, B, and G
matrix_size = 256 * 256
A = parse_matrix(data[:matrix_size])
B = parse_matrix(data[matrix_size:2 * matrix_size])
G = parse_matrix(data[2 * matrix_size:3 * matrix_size])

# Get image dimensions
m, n = G.shape


# Display the image
plt.imshow(G, cmap='gray')
plt.title('Image G')
plt.imsave('./images_results/Image_G.png', G, cmap='gray')
# plt.show()

# Compute the Singular Value Decomposition of A and B
Ua, sa, Vha = linalg.svd(A)
Ub, sb, Vhb = linalg.svd(B)

# Compute G
G_new = np.dot(np.dot(Ub.T, G), Ua)

# Construct S matrix using the outer product of the singular values
S = np.outer(sb, sa)

# Define the alpha values for Tikhonov regularization
start_value = 0.5
# Define the factor by which the alpha values will be multiplied
factor = 0.5
# Adjust the number of elements as needed, if you want test more values
num_elements = 10
alpha_test = [start_value * (factor ** i) for i in range(num_elements)]

# In case you want to test only one value of alpha
# alpha_test = [0.0005]

# # Iterate over alpha values
for alpha in alpha_test:
    # Compute Tikhonov regularization
    F_new = (S * G_new) / (S * S + alpha**2)
    F = np.dot(np.dot(Vhb.T, F_new), Vha)

    # Display reconstructed image
    plt.imshow(F, cmap='gray')
    plt.title(f'Tikhonov Regularization, alpha = {alpha}')
    plt.imsave(f'./images_results/Tikhonov_alpha_{alpha}.jpg', F, cmap='gray')
    # plt.show()


# TSVD

# Reshape S to get 1D array
# Store the entries of S in a linear array s, then sort this set in descending order
s = np.reshape(S, -1)
# Sorting the array and get the sorted indices
prm = np.argsort(s)[::-1]  # Reverse to sort in descending order
ss = np.sort(s)[::-1]

# Using the sorted set and permutation constructed above
ls = len(s)

iprm = np.zeros((ls, 1), dtype=int)
iprm[prm] = np.arange(ls)[:, np.newaxis]

# set of values of p.
start_v = 100
step_size = 350
pset = np.arange(start_v, (num_elements+1) * step_size, step_size)

# In case you want to test only one value of p
# pset = [2500]

for p in pset:
    ssnew = np.concatenate((ss[:p], np.zeros(ls - p)))
    Snew = np.reshape(ssnew[iprm], (n, n))
    Fhat = np.divide(G_new, S)
    Fnew = Fhat * (Snew > 0)  # multiply by only Snew > than 0
    
    # Reconstruct the image
    F = np.dot(np.dot(Vhb.T, Fnew), Vha)
    plt.imshow(F, cmap='gray')
    plt.title(f'TSVD p={p}')
    plt.imsave(f'./images_results/TSVD_p_{p}.jpg', F, cmap='gray')
    # plt.show()