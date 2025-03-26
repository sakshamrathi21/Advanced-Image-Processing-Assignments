import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.ndimage import rotate as scipy_rotate
from skimage import data
from skimage.io import imread
from skimage.transform import radon, resize, iradon, rotate
from skimage.metrics import mean_squared_error

def add_noise_to_radon(R, f=5):
    """
    Adds Gaussian noise to Radon projections.

    Parameters:
    - R: numpy array, Radon projections (2D array)
    - f: float, fraction for sigma calculation (default: 0.05)

    Returns:
    - noisy_R: numpy array, Radon projections with added noise
    """
    f = f*0.01
    # print(np.mean(np.abs(R)))
    sigma = f * np.mean(np.abs(R))  # Calculate noise based on mean of absolute values
    # print("Sigma of noise is", sigma)
    noisy_R = R + np.random.normal(0, sigma, R.shape)
    noisy_R = np.clip(noisy_R, 0, np.max(noisy_R))
    return noisy_R, sigma
def add_noise_to_radon_dB(R, dB = 3):
    """
    Adds Gaussian noise to Radon projections.

    Parameters:
    - R: numpy array, Radon projections (2D array)
    - f: float, fraction for sigma calculation (default: 0.05)

    Returns:
    - noisy_R: numpy array, Radon projections with added noise
    """
    variance_signal = np.var(R)
    # print(variance_signal)
    noise_power = variance_signal / (10 ** (dB / 10))  # Calculate noise power based on SNRdB
    # print("shape of variance", variance_signal.shape)
    sigma = np.sqrt(noise_power)
    # sigma = f * np.mean(np.abs(R))  # Calculate noise based on mean of absolute values
    # print("Sigma of noise is", sigma)
    noisy_R = R + np.random.normal(0, sigma, R.shape)
    noisy_R = np.clip(noisy_R, 0, np.max(noisy_R))
    return noisy_R, sigma
def generate_samples(img_path, num_angles=500, distribution='uniform', ideal_method = False, use_phantom = False, noise = False, noise_percentage = 5, circle = False):
    # Load the Phantom image
    # Load the Shepp-Logan Phantom image
    if use_phantom:
        img_gray = data.shepp_logan_phantom()
    # Load the image
    img = imread(img_path)
    if use_phantom:
      img_gray = resize(data.shepp_logan_phantom(), (512, 512), anti_aliasing=True)

    else:
      img_gray = img
    # # Convert to grayscale if necessary
    #   img_gray = 1 - resize(rgb2gray(img), (512, 512), anti_aliasing=True)
    # img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    img_gray = img_gray.astype(np.float64)  # Ensure float type before division
    img_gray /= np.sum(img_gray)
    img_gray *= 100


    if distribution == 'uniform':
        # Generate random angles uniformly distributed
        theta = (np.random.rand(num_angles)- 0.5) * 360
        # theta = np.concatenate((theta, -theta))
    elif distribution == 'uniform_spaced':
        # Generate angles uniformly spaced from -180 to 180
        theta = np.linspace(-180, 180, num_angles)
        np.random.shuffle(theta)  # Shuffle the angles randomly

    # Perform the Radon transform
    R = radon(img_gray, theta=theta, circle=circle, preserve_range= True)
    noisy_R = R.copy()
    sigma = 0
    if(noise == True):
        # noisy_R, sigma = add_noise_to_radon(R, f = noise_percentage)
        noisy_R, sigma = add_noise_to_radon_dB(R, dB = noise_percentage)
    if ideal_method == True:
        # Sort R and theta based on theta
        theta_sorted = theta.copy()
        sorted_indices = np.argsort(theta_sorted)
        first_index = sorted_indices[0]
        second_index = sorted_indices[1]
        # print(sorted_indices)
        R_ordered = R.copy()
        R_ordered = R_ordered[:, sorted_indices]
        theta_sorted = theta_sorted[sorted_indices]

    # Plot the original image
    plt.figure()
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image')
    plt.savefig("Original Image.png")
    plt.show()


    plt.figure()
    plt.imshow(np.clip(iradon(noisy_R, theta=theta, circle=circle, preserve_range=True), 0, 1), cmap='gray')
    plt.title('Reconstructed Image')
    plt.savefig(f"ReconstructedImage_noisedB_{noise_percentage}_numprojs_{num_angles}.png")
    plt.show()

    # Plot the Radon transform
    plt.figure()
    plt.imshow(noisy_R, extent=(min(theta), max(theta), 0, R.shape[0]), aspect='auto', cmap='jet')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Projection axis')
    plt.colorbar(label='Intensity')
    plt.title('Radon Transform - Randomly Sampled Projections')
    plt.show()

    # Plot histogram of theta
    plt.figure()
    plt.hist(theta, bins=360)  # Adjust the number of bins as needed
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Angles for Radon Transform')
    plt.show()

    return img_gray, R, sorted_indices, theta_sorted, theta, first_index, second_index, R_ordered, noisy_R, sigma
def custom_kernel_matrix(X, epsilon, noise_levels=0):
    pairwise_distances = cdist(X, X, 'sqeuclidean')
    N = len(X)

    # Set epsilon based on noise levels
    if noise_levels == 0:
        epsilon = 0.025
    elif noise_levels == 40:
        epsilon = 0.03
    elif noise_levels == 20:
        epsilon = np.sort(pairwise_distances[0])[N // 100]
    elif noise_levels == 15:
        epsilon = np.sort(pairwise_distances[0])[N // 75]
    elif noise_levels in [25, 30]:
        epsilon = np.sort(pairwise_distances[0])[N // 150]
    
    # print(f"Selected epsilon: {epsilon}")

    # Compute the kernel matrix using the Gaussian kernel
    kernel_matrix = np.exp(-pairwise_distances / (2 * epsilon))
    
    # Zero out elements where distance is greater than epsilon
    kernel_matrix[pairwise_distances > epsilon] = 0

    return kernel_matrix, epsilon

def compute_W_sum(args):
    X, epsilon_val = args
    W, _ = custom_kernel_matrix(X, epsilon_val)
    return epsilon_val, np.sum(W)

def laplacian_eigenmapping_custom_kernel_optimized(X, noise_levels=0, epsilon=0.01, n_components=2, 
                                                    epsilon_range=None, if_plot=True):
    if epsilon_range is None:
        epsilon_range = [1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10, 13, 15]
    
    N = len(X)
    np.set_printoptions(precision=20)

    if if_plot:
        args_list = [(X, epsilon_val) for epsilon_val in epsilon_range]
        with multiprocessing.Pool() as pool:
            results = pool.map(compute_W_sum, args_list)
        epsilon_range, W_sum = zip(*results)
        
        plt.figure()
        plt.plot(np.log10(np.array(epsilon_range)), np.log10(W_sum))
        plt.show()
    
    W, epsilon = custom_kernel_matrix(X, epsilon, noise_levels=noise_levels)
    
    D = np.diag(np.sum(W, axis=1, dtype=np.float64))
    L = D - W
    
    D_inv = np.diag(1 / np.sum(W, axis=1, dtype=np.float64))
    W_tilde = np.dot(D_inv, np.dot(W, D_inv))
    
    D_tilde = np.diag(np.sum(W_tilde, axis=1, dtype=np.float64))
    L_prime = D_tilde - W_tilde
    
    eigenvalues, eigenvectors = eigh(L_prime, D_tilde, subset_by_index=[1, n_components])
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    
    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:", eigenvectors)
    
    return eigenvectors.T, W
from PIL import Image
import numpy as np
from skimage.transform import rotate

def mse_rotation(angle, img1, img2):
    """Calculate the error between two images after rotation using the given formula."""
    rotated_img = rotate(img1, angle, resize=False)
    numerator = np.mean((rotated_img - img2) ** 2)
    denominator = np.sum(np.abs(img2) ** 2)
    mse = numerator / denominator
    return mse, rotated_img

def find_best_rotation_angle(recon_om, iom):
    """
    Find the rotation angle that aligns the reconstructed image with the original image.

    Args:
    - recon_om (numpy.ndarray): The reconstructed image.
    - iom (numpy.ndarray): The original image.

    Returns:
    - best_rotation_angle (int): The best rotation angle that minimizes the error.
    """
    recon_om_orignal = recon_om.copy()
    recon_om_orignal_mirror = np.flipud(recon_om_orignal)
    recon_om = np.clip(recon_om, 0, 1)
    recon_om = (recon_om - np.min(recon_om))/(np.max(recon_om) - np.min(recon_om))
    iom = (iom - np.min(iom))/(np.max(iom) - np.min(iom))
    # Define a range of rotation angles to explore
    rotation_angles = range(-180, 181)

    # Initialize variables to store the best rotation angle and its corresponding error
    best_rotation_angle = 0
    mirror_image = np.flipud(recon_om)
    min_error = float('inf')
    best_rotated_image = recon_om
    # Visual inspection: Display images for each rotation angle and compute error
    for angle in rotation_angles:
        # Compute error using the custom formula for the original rotation
        error, rotated_img = mse_rotation(angle, recon_om, iom)
        if error < min_error:
          min_error = error
          best_rotation_angle = angle
          best_rotated_image = rotate(recon_om_orignal, best_rotation_angle, resize=False)
        # Compute error using the custom formula for the mirrored rotation
        mirrored_error, rotated_img = mse_rotation(angle, mirror_image, iom)
        # print(mirrored_error, min_error)
        if mirrored_error < min_error:
          min_error = mirrored_error
          best_rotation_angle = angle
          best_rotated_image = rotate(recon_om_orignal_mirror, best_rotation_angle, resize=False)
          # plt.imshow(best_rotated_image, cmap = 'gray')
          # plt.show()
          # print(mirrored_error)

        # Choose the minimum error between original and mirrored rotation
        # min_rotation_error = min(error, mirrored_error)
        # print(min_rotation_error)
        # Update best rotation angle if error is minimized



    # Print the best rotation angle
    print("Best Rotation Angle:", best_rotation_angle)

    return min_error, best_rotated_image

distribution = 'uniform'
num_angles = 5000
circle = True
test_num_projs = [50, 100, 500, 1000, 2000, 5000, 10000]
# test_num_projs = [5000, 7000, 10000, 15000]
# test_num_projs = [7000]
min_error_for_projections = {}
for num_angles in test_num_projs:
    noise_percentage = 0
    img_gray, R, ideal_sorted_indices, theta_sorted, theta_org, first_index, second_index, R_ordered, noisy_R, sigma = generate_samples('./cryoem.png', num_angles = num_angles, distribution=distribution, ideal_method= True, use_phantom = False, noise = False, noise_percentage = noise_percentage, circle=circle)
    embedded_data_custom_kernel_normalized, W = laplacian_eigenmapping_custom_kernel_optimized(R.T, epsilon = 0.02, n_components=2, if_plot = False, noise_levels=noise_percentage)
    plt.figure()
    plt.scatter(embedded_data_custom_kernel_normalized[0], embedded_data_custom_kernel_normalized[1])
    plt.savefig(f"embedded_data_custom_kernel_normalized_{num_angles}_{noise_percentage}dB.png")
    plt.show()
    embedding = embedded_data_custom_kernel_normalized

    # Calculate the angles (arctangent) for each embedding
    angles_rad = np.arctan2(embedding[1], embedding[0])
    # angles_rad += 2 * np.pi * (angles_rad < 0)
    ordered_indices = np.argsort(angles_rad)
    num_embeddings = len(ordered_indices)
    angles_unif = np.linspace(-180, 180, num_embeddings, endpoint=False)
    ordered_angles_unif = np.linspace(-180, 180, num_embeddings, endpoint=False)
    ordered_angles_unif[ordered_indices] = angles_unif
    reconstructed_image = iradon(R, theta=ordered_angles_unif, circle = circle, preserve_range = True)
    min_error, best_rotated_image = find_best_rotation_angle(reconstructed_image, img_gray)
    min_error_formatted = f"{min_error:.2e}"  # Displays min_error in scientific notation with 2 decimal places
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(best_rotated_image, cmap='gray')
    plt.title("Reconstructed Image")  # Title for the first subplot
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Image")  # Title for the second subplot
    
    if(noise_percentage == 0):
        plt.suptitle(f"Noiseless {num_angles} Projections RRMSE {min_error_formatted}", ha='center')
        plt.savefig(f"Recon_original_Noiseless_{num_angles}_rrmse_{min_error_formatted}.png")
    else:
        plt.suptitle(f"{noise_percentage}dB {num_angles} Projections RRMSE {min_error_formatted}", ha='center')
        plt.savefig(f"Recon_original_{noise_percentage}db_{num_angles}_rrmse_{min_error_formatted}.png")        
    plt.show()
    min_error_for_projections[num_angles] = min_error
plt.figure()
plt.plot(min_error_for_projections.keys(), min_error_for_projections.values())
plt.title("RRMSE vs Number of Projections")
plt.savefig(f"RRMSE_NumProjs.png")
plt.show()
