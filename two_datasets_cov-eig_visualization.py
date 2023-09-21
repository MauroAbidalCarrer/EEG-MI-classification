from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import namedtuple

ARROW_WIDTH = 0.003

def generate_synthetic_data(num_samples, variances):
    return np.random.multivariate_normal((0, 0), np.diag(variances), num_samples)

def rotation_matrix_2d(theta):
    theta = math.radians(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

def draw_rotated_dataset(data, rotation_angle, dataset_colors, data_index):
    #Data generation.
    rotation_matrix = rotation_matrix_2d(rotation_angle)
    roatated_data = data @ rotation_matrix
    cov_matrix = np.cov(roatated_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    #Data visualization.
    ax.scatter(roatated_data[:, 0], roatated_data[:, 1], c=dataset_colors.scatter, alpha=0.35, label=f'dataset {data_index}')
    ax.quiver(0, 0, cov_matrix[0, 1], cov_matrix[1, 1], angles='xy', scale_units='xy', scale=1, color=dataset_colors.cov, label=f'Covariance Vectors {data_index}', width=ARROW_WIDTH)
    ax.quiver(0, 0, cov_matrix[0, 0], cov_matrix[1, 0], angles='xy', scale_units='xy', scale=1, color=dataset_colors.cov, width=ARROW_WIDTH)
    for i in range(len(eigenvalues)):
        eigen_vector = eigenvectors[:, i]
        scaled_eigen_vector = eigenvalues[i] * eigen_vector
        ax.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color=dataset_colors.eig, label=f'Covariance Eigenvectors {data_index}', width=ARROW_WIDTH)
    ax.axis('equal')
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

def update(_):
    ax.clear()
    for i in range(len(datasets)):
        draw_rotated_dataset(datasets[i], sliders[i].val, datasets_colors[i], i)
    plt.draw()


datasets = [generate_synthetic_data(500, (8, 1.5)),  generate_synthetic_data(500, (9, 2.5))]
Dataset_colors = namedtuple('Dataset_colors', 'scatter cov eig')
datasets_colors = [Dataset_colors('blue', 'green', 'magenta'), Dataset_colors('red', 'pink', 'black')]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Sliders initialization
SLIDERS_TOP = 0.15
ax_sliders = [plt.axes([0.25, SLIDERS_TOP - 0.05 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow') for i in range(len(datasets))]
sliders = [Slider(ax_slider, f'Rotation Angle {i}', 0, 360, valinit=0) for i, ax_slider in enumerate(ax_sliders)]
for slider in sliders:
    slider.on_changed(update)

update(0)
plt.show()
