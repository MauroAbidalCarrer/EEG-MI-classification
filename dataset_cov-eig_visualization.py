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

def update(_):
    rotation_matrix = rotation_matrix_2d(rotation_slider.val)
    data = original_data @ rotation_matrix
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Calculate the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Clear the previous plot
    og_plot.clear()
    
    # Plot the data points
    og_plot.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.4)

    # Plot the covariance column vectors in green
    og_plot.quiver(0, 0, cov_matrix[0, 0], cov_matrix[1, 0], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 1')
    og_plot.quiver(0, 0, cov_matrix[0, 1], cov_matrix[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 2')

    # Plot the eigenvectors as red arrows
    for i in range(len(eigenvalues)):
        eigen_vector = eigenvectors[:, i]
        scaled_eigen_vector = eigenvalues[i] * eigen_vector
        og_plot.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'Eigenvector {i+1}')

    plot_side_length = np.amax(np.linalg.norm(data, axis=1)) * 1.1
    og_plot.axis([-plot_side_length, plot_side_length, -plot_side_length, plot_side_length])
    og_plot.set_aspect(1)

    cov_dot_plot.clear()
    cov_projected_data = data @ cov_matrix
    t = interpolation_slider.val
    interpolated_data = np.array([data[i] * (t - 1) + cov_projected_data[i] * t for i in range(data.shape[0])])
    cov_dot_plot.scatter(interpolated_data[:, 0], interpolated_data[:, 1], c='blue', alpha=0.3)
    plot_side_length = np.amax(np.linalg.norm(interpolated_data, axis=1)) * 1.1
    cov_dot_plot.axis([-plot_side_length, plot_side_length, -plot_side_length, plot_side_length])
    cov_dot_plot.set_aspect(1)

    plt.draw()

original_data = generate_synthetic_data(1000, (8, 3))

fig, axes = plt.subplots(1, 2)
og_plot = axes[0]
cov_dot_plot = axes[1]

plt.subplots_adjust(bottom=0.25)

# Create a slider for the rotation angle
def mk_slider(bottom, name, max_value):
    ax_slider = plt.axes([0.25, bottom, 0.5, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, name, 0, max_value, valinit=0)
    slider.on_changed(update)
    return slider

rotation_slider = mk_slider(0.1, 'Rotation angle', 360)
interpolation_slider = mk_slider(0, 'interpolation', 1)

update(0)
plt.show()
