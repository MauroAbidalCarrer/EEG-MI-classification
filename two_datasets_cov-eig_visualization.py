from collections import OrderedDict
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider
from collections import namedtuple
from scipy.linalg import eigh

ARROW_WIDTH = 0.03

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

# def draw_rotated_dataset(data, rotation_angle, dataset_colors, data_index):
#     #Data generation.
#     cov_matrix = np.cov(rotated_data, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#     #Data visualization.
#     ax.quiver(0, 0, cov_matrix[0, 1], cov_matrix[1, 1], angles='xy', scale_units='xy', scale=1, color=dataset_colors.cov, label=f'Covariance Vectors {data_index}', width=ARROW_WIDTH)
#     ax.quiver(0, 0, cov_matrix[0, 0], cov_matrix[1, 0], angles='xy', scale_units='xy', scale=1, color=dataset_colors.cov, width=ARROW_WIDTH)
#     for i in range(len(eigenvalues)):
#         eigen_vector = eigenvectors[:, i]
#         scaled_eigen_vector = eigenvalues[i] * eigen_vector
#         ax.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color=dataset_colors.eig, label=f'Covariance Eigenvectors {data_index}', width=ARROW_WIDTH)
#     ax.axis('equal')
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = OrderedDict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys())

def update(_):
    og_plot.clear()
    rotated_datasets = [rotated_matrix(datasets[i], sliders[i].val) for i in range(len(datasets))]
    for i in range(len(datasets)):
        og_plot.scatter(rotated_datasets[i][:, 0], rotated_datasets[i][:, 1], c=datasets_colors[i].scatter, alpha=0.2, label=f'dataset {i}')

    cov_matrices = np.array([rotated_data.T @ rotated_data for rotated_data in rotated_datasets])
    # def visulize_cov_eig_vecs(cov_matrix):
    #     cov_eig_vals, cov_eig_vecs = eig(cov_matrix)
    #     for i in range(len(cov_eig_vals)):
    #         scaled_eigen_vector = cov_eig_vecs[:, i].real * math.log(cov_eig_vals[i].real) * 2
    #         og_plot.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='red')
    # for cov_matrix in cov_matrices:
    #     visulize_cov_eig_vecs(cov_matrix)

    gen_eig_vals, gen_eig_vecs = eigh(cov_matrices[0], cov_matrices.sum(0))
    gen_eig_vals = gen_eig_vals.real
    gen_eig_vecs = gen_eig_vecs.real
    for i in range(len(gen_eig_vals)):
        vec = gen_eig_vecs[:, i]
        scaled_eigen_vector = vec * math.log(gen_eig_vals[i]) * 2
        # scaled_eigen_vector = vec * 2
        og_plot.quiver(0, 0, scaled_eigen_vector[0], scaled_eigen_vector[1], angles='xy', scale_units='xy', scale=1, color='magenta')
        

    plot_side_length = np.amax([np.linalg.norm(dataset, axis=1) for dataset in datasets]) * 1.1
    og_plot.axis([-plot_side_length, plot_side_length, -plot_side_length, plot_side_length])
    og_plot.set_aspect(1)

    fil_plot.clear()
    # sort_oreder = np.argsort(gen_eig_vals)[::-1]
    # gen_eig_vecs = np.array([normalize_vector(gen_eig_vec) for gen_eig_vec in gen_eig_vecs]) #[sort_oreder]
    plot_side_length /= 50
    for i, data in enumerate(rotated_datasets):
        projected_data = data @ gen_eig_vecs
        fil_plot.scatter(projected_data[:, 0], projected_data[:, 1], c=datasets_colors[i].scatter, alpha=0.2)
    fil_plot.axis([-plot_side_length, plot_side_length, -plot_side_length, plot_side_length])
    fil_plot.set_aspect(1)

    plt.draw()


datasets = [generate_synthetic_data(250, (8, 1)),  generate_synthetic_data(250, (9, 1))]
Dataset_colors = namedtuple('Dataset_colors', 'scatter cov eig')
datasets_colors = [Dataset_colors('red', 'green', 'magenta'), Dataset_colors('blue', 'pink', 'black')]

# fig, axes = plt.subplots(1, 2)
# og_plot = axes[0]
# fil_plots = axes[1].subplots(2, 1)
# pool_fil_plot = fil_plots[0]
# coventianal_cov_fil_plot = fil_plots[1]

# Create the main subplot with a 1x2 grid
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, figure=fig)

# Create the first subplot in the main subplot (gs[0])
og_plot = fig.add_subplot(gs[0])

# Create a nested subplot in the second column (gs[1])
gs_nested = GridSpec(2, 1, figure=fig, left=0.6, right=0.98, top=0.9, bottom=0.1)
pool_fil_plot = fig.add_subplot(gs_nested[0])

coventianal_cov_fil_plot = fig.add_subplot(gs_nested[1])

# # Adjust the layout for better spacing
# plt.tight_layout()

plt.subplots_adjust(bottom=0.25)

# Sliders initialization
SLIDERS_TOP = 0.15
ax_sliders = [plt.axes([0.25, SLIDERS_TOP - 0.05 * i, 0.65, 0.03], facecolor='lightgoldenrodyellow') for i in range(len(datasets))]
sliders = [Slider(ax_slider, f'Rotation Angle {i}', 0, 360, valinit=0) for i, ax_slider in enumerate(ax_sliders)]
for slider in sliders : slider.on_changed(update)

update(0)
plt.show()
