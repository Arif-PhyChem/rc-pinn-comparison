import numpy as np
import matplotlib.pyplot as plt


fmo_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/sr_traj.npz', 'SR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/psr_traj.npz', 'PSR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/mr_traj.npz', 'MR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/pmr_traj.npz', 'PMR-PINN')
]

# Reference data for FMO complex
fmo_ref = np.load('/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/test_data/init_1/7_initial-1_gamma-400.0_lambda-40.0_temp-90.0.npy')
fmo_ref_time = fmo_ref[:, 0] / 1000  # Convert time to ps
fmo_diag_indices = [1, 9, 17, 25, 33, 41, 49]  # Diagonal element indices
fmo_offdiag_indices = [2, 10, 18]  # Off-diagonal elements: rho_12, rho_23, rho_34

# Create a 3x3 figure
fig, axes = plt.subplots(4, 2, figsize=(6, 8), constrained_layout=True, sharex='col')

# Colors for plots
colors_diag = ['blue', 'green']  # For SB diagonal
colors_offdiag_sb = ['red', 'orange']  # For SB off-diagonal
colors_fmo = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']  # For FMO
colors_fmo_re_offdiag = ['blue', 'green', 'red'] # For FMO real off-diagonal elements
colors_fmo_im_offdiag = ['orange', 'purple', 'brown'] # For FMO imag off-diagonal elements

# Plot FMO diagonal elements (Second Column)
for row_idx, (file, label) in enumerate(fmo_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Diagonal elements
    re_rho_diag = [np.real(rho[:, i, i]) for i in range(7)]
    ref_re_rho_diag = [fmo_ref[:, idx] for idx in fmo_diag_indices]

    ax = axes[row_idx, 0]
    for i, (diag, ref_diag, color) in enumerate(zip(re_rho_diag, ref_re_rho_diag, colors_fmo)):
        ax.plot(time_pred, diag, color=color, label=f"$\\rho_{{\\mathrm{{S}},{i+1}{i+1}}}$")
        ax.plot(fmo_ref_time, ref_diag, 'o', color=color, markevery=20, markersize=4)

    ax.set_title(f"FMO: {label}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.2, color='black', linestyle='--')
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
    #    ax.get_legend().remove()  # Remove legend from all other rows

# Plot FMO off-diagonal elements (Third Column)
for row_idx, (file, label) in enumerate(fmo_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Off-diagonal elements: rho_12, rho_23, rho_34
    re_rho_offdiag = [np.real(rho[:, 0, 1]), np.real(rho[:, 1, 2]), np.real(rho[:, 2, 3])]
    im_rho_offdiag = [np.imag(rho[:, 0, 1]), np.imag(rho[:, 1, 2]), np.imag(rho[:, 2, 3])]
    ref_re_offdiag = [np.real(fmo_ref[:, idx]) for idx in fmo_offdiag_indices]
    ref_im_offdiag = [np.imag(fmo_ref[:, idx]) for idx in fmo_offdiag_indices]

    ax = axes[row_idx, 1]
    labels_real = [r"Re($\rho_{\mathrm{S},12}$)", r"Re($\rho_{\mathrm{S},23}$)", r"Re($\rho_{\mathrm{S},34}$)"]
    labels_imag = [r"Im($\rho_{\mathrm{S},12}$)", r"Im($\rho_{\mathrm{S},23}$)", r"Im($\rho_{\mathrm{S},34}$)"]

    for i, (re_rho, ref_re, label_re, color) in enumerate(
        zip(re_rho_offdiag, ref_re_offdiag, labels_real, colors_fmo_re_offdiag)
    ):
        ax.plot(time_pred, re_rho, color=color, label=label_re)
        ax.plot(fmo_ref_time, ref_re, 'o', color=color, markevery=15, markersize=4)
    
    for i, (im_rho, ref_im, label_im, color) in enumerate(
        zip(im_rho_offdiag, ref_im_offdiag, labels_imag, colors_fmo_im_offdiag)
    ):
        ax.plot(time_pred, im_rho, color=color, label=label_im)
        ax.plot(fmo_ref_time, ref_im, 'o', color=color, markevery=15, markersize=4)

    ax.set_title(f"FMO: {label}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.2, color='black', linestyle='--')
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
    #    ax.get_legend().remove()  # Remove legend from all other rows

#

#fig.supylabel('Density Matrix Elements', fontsize=14)

# Adjust labels for each column
# Second and third columns: FMO complex with time in ps
axes[3, 0].set_xlabel('Time (ps)', fontsize=12)
axes[3, 1].set_xlabel('Time (ps)', fontsize=12)


axes[0, 0].text(-0.20, 1.20, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.20, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.20, 1.20, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 0].text(-0.20, 1.20, 'D', transform=axes[3, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.20, 1.20, 'E', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.20, 'F', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.20, 1.20, 'G', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 1].text(-0.20, 1.20, 'H', transform=axes[3, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')


# Save and show the plot
plt.savefig('fmo_1_dyn.pdf', format='pdf', dpi=300)
plt.show()

