import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import namedtuple
from scipy.linalg import eigh, eig, norm, inv

ARROW_WIDTH = 0.03

def sigmoid(x):
 return 1/(1 + np.exp(-x))


def generate_synthetic_data(num_samples, variances):
    return np.random.multivariate_normal((0, 0), np.diag(variances), num_samples)

def rotated_matrix(matrix, angle):
    theta = math.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return matrix @ np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector  # Avoid division by zero
    return vector / norm

def spatialFilter(Ra,Rb):
    def sorted_eig(vals, vecs):
        ord = np.argsort(vals)[::-1]
        return vals[ord].real, vecs[:, ord].real

    # Compute whitening matrix.
    E, U = sorted_eig(*eig(Ra + Rb))
    white_mat = np.sqrt(inv(np.diag(E))) @ U.T
    white_mat = white_mat.real

    # The mean covariance matrices may now be transformed
    Sa = white_mat @ Ra @ white_mat.T
    Sb = white_mat @ Rb @ white_mat.T

    # Find and sort the generalized eigenvalues and eigenvector
    _, U1 = sorted_eig(*eig(Sa, Sb))

    # The projection matrix (the spatial filter) may now be obtained
    SFa = U1.T @ white_mat
    return SFa.astype(np.float32)

def set_plot_side_length(plot, side_length):
    plot.axis([-side_length, side_length, -side_length, side_length])
    plot.set_aspect(1)

def update(_):
    og_plot.clear()
    fil_plot.clear()

    rotated_datasets = [rotated_matrix(datasets[i], sliders[i].val) for i in range(len(datasets))]
    for i in range(len(datasets)):
        og_plot.scatter(rotated_datasets[i][:, 0], rotated_datasets[i][:, 1], c=datasets_colors[i].scatter, alpha=0.2, label=f'dataset {i}')


    cov_matrices = np.array([rotated_data.T @ rotated_data for rotated_data in rotated_datasets])
    filters = spatialFilter(cov_matrices[0], cov_matrices[1])

    #Visualize filters
    for i in range(len(filters)):
        vec = filters[:, i]
        scaled_eigen_vector = vec / norm(vec)
        og_plot.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='green', label='filters')

    for i, data in enumerate(rotated_datasets):
        projected_data = data @ filters
        fil_plot.scatter(projected_data[:, 0], projected_data[:, 1], c=datasets_colors[i].scatter, alpha=0.2)
    
    #set dims
    plot_side_length = np.amax([np.linalg.norm(dataset, axis=1) for dataset in datasets]) * 1.1
    set_plot_side_length(og_plot, plot_side_length)
    set_plot_side_length(fil_plot, plot_side_length / 25)
    plt.draw()


datasets = [generate_synthetic_data(250, (10, 1)),  generate_synthetic_data(250, (10, 1))]
Dataset_colors = namedtuple('Dataset_colors', 'scatter cov eig')
datasets_colors = [Dataset_colors('red', 'green', 'magenta'), Dataset_colors('blue', 'pink', 'black')]

fig, axes = plt.subplots(1, 2)
og_plot = axes[0]
fil_plot = axes[1]
plt.subplots_adjust(bottom=0.25)

# Sliders initialization
SLIDERS_TOP = 0.15
ax_sliders = [plt.axes([0.25, SLIDERS_TOP - 0.05 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow') for i in range(len(datasets))]
sliders = [Slider(ax_slider, f'Rotation Angle {i}', 0, 360, valinit=0) for i, ax_slider in enumerate(ax_sliders)]
for slider in sliders : slider.on_changed(update)

update(0)
plt.show()