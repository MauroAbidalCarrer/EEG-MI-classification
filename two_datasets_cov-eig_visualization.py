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

def draw_rotated_dataset(data, rotation_angle, color):
    rotation_matrix = rotation_matrix_2d(rotation_angle)
    roatated_data = data @ rotation_matrix
    
    cov_matrix = np.cov(roatated_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    ax.scatter(roatated_data[:, 0], roatated_data[:, 1], c=color, alpha=0.2)

    ax.quiver(0, 0, cov_matrix[0, 0], cov_matrix[1, 0], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 1')
    ax.quiver(0, 0, cov_matrix[0, 1], cov_matrix[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Covariance Vector 2')
    for i in range(len(eigenvalues)):
        eigen_vector = eigenvectors[:, i]
        scaled_eigen_vector = eigenvalues[i] * eigen_vector
        ax.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'Eigenvector {i+1}')

    ax.axis('equal')

def update(_):
    ax.clear()
    for i in range(len(datasets)):
        draw_rotated_dataset(datasets[i], sliders[i].val, datasets_colors[i])
    plt.draw()

datasets = [generate_synthetic_data(1000, (8, 3)),  generate_synthetic_data(500, (9, 2.5))]
datasets_colors = ['blue', 'magenta']

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Create sliders for the rotation angles
sliders_ax_rects = [[0.25, 0.15, 0.65, 0.03], [0.25, 0.1, 0.65, 0.03]]
ax_sliders = [plt.axes(rect, facecolor='lightgoldenrodyellow') for rect in sliders_ax_rects]
sliders = [Slider(ax_slider, f'Rotation Angle {i}', 0, 360, valinit=0) for i, ax_slider in enumerate(ax_sliders)]
for slider in sliders:
    slider.on_changed(update)

update(0)
plt.show()
