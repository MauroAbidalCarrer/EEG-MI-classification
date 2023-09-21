import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def generate_synthetic_data(num_samples, variances):
    cov_matrix = np.diag(variances)  # Create a covariance matrix with the specified variances
    
    # Generate random samples using the specified mean and covariance matrix
    synthetic_data = np.random.multivariate_normal((0, 0), cov_matrix, num_samples)
    
    return synthetic_data

def rotation_matrix_2d(theta):
    theta = math.radians(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Create the 2D rotation matrix
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    
    return rotation_matrix

def update(rotation_angle):
    rotation_matrix = rotation_matrix_2d(rotation_angle)
    data = original_data @ rotation_matrix
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Calculate the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Clear the previous plot
    ax.clear()
    
    # Plot the data points
    ax.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.4)

    # Plot the covariance column vectors in green
    ax.quiver(0, 0, cov_matrix[0, 0], cov_matrix[1, 0], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 1')
    ax.quiver(0, 0, cov_matrix[0, 1], cov_matrix[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 2')

    # Plot the eigenvectors as red arrows
    for i in range(len(eigenvalues)):
        eigen_vector = eigenvectors[:, i]
        scaled_eigen_vector = eigenvalues[i] * eigen_vector
        ax.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'Eigenvector {i+1}')

    ax.axis('equal')
    plt.draw()

original_data = generate_synthetic_data(1000, (8, 3))

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Create a slider for the rotation angle
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Rotation Angle', 0, 360, valinit=0)

slider.on_changed(update)
update(slider.val)
plt.show()
