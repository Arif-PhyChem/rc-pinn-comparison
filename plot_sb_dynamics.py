import numpy as np
import matplotlib.pyplot as plt

# File names and labels
sym_sb_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/sr_traj.npz', 'SR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/psr_traj.npz', 'PSR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/mr_traj.npz', 'MR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/pmr_traj.npz', 'PMR-PINN')
]
# File names and labels
asym_sb_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/sr_traj.npz', 'SR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/psr_traj.npz', 'PSR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/mr_traj.npz', 'MR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/pmr_traj.npz', 'PMR-PINN')
]

# Reference data for symetric SB model
sym_sb_ref = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-0.0_Delta-1.0_lambda-0.6_gamma-3.0_beta-1.0.npy')
sym_sb_ref_time = np.real(sym_sb_ref[:, 0])  # Reference time

# Reference data for asymetric SB model
asym_sb_ref = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-1.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy')
asym_sb_ref_time = np.real(asym_sb_ref[:, 0])  # Reference time


# Create a 4x2 figure
fig, axes = plt.subplots(4, 2, figsize=(6, 8), constrained_layout=True, sharex='col')

# Colors for plots
colors_diag = ['blue', 'green']  # For SB diagonal
colors_offdiag_sb = ['red', 'orange']  # For SB off-diagonal
colors_fmo = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']  # For FMO
colors_fmo_re_offdiag = ['blue', 'green', 'red'] # For FMO real off-diagonal elements
colors_fmo_im_offdiag = ['orange', 'purple', 'brown'] # For FMO imag off-diagonal elements

# Plot SB model (First Column)
for row_idx, (file, label) in enumerate(sym_sb_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Diagonal elements
    re_rho_11 = np.real(rho[:, 0, 0])
    re_rho_22 = np.real(rho[:, 1, 1])
    sym_sb_ref_re_rho_11 = np.real(sym_sb_ref[:, 1])
    sym_sb_ref_re_rho_22 = np.real(sym_sb_ref[:, 4])

    # Off-diagonal elements
    re_rho_12 = np.real(rho[:, 0, 1])
    im_rho_12 = np.imag(rho[:, 0, 1])
    sym_sb_ref_re_rho_12 = np.real(sym_sb_ref[:, 2])
    sym_sb_ref_im_rho_12 = np.imag(sym_sb_ref[:, 2])

    ax = axes[row_idx, 0]
    ax.plot(time_pred, re_rho_11, color=colors_diag[0], label="$\\rho_{\mathrm{S},11}$")
    ax.plot(time_pred, re_rho_22, color=colors_diag[1], label="$\\rho_{\mathrm{S},22}$")
    ax.plot(sym_sb_ref_time, sym_sb_ref_re_rho_11, 'o', color=colors_diag[0], markevery=20, markersize=4)
    ax.plot(sym_sb_ref_time, sym_sb_ref_re_rho_22, 'o', color=colors_diag[1], markevery=20, markersize=4)

    ax.plot(time_pred, re_rho_12, color=colors_offdiag_sb[0], label="Re($\\rho_{\mathrm{S},12}$)")
    ax.plot(time_pred, im_rho_12, color=colors_offdiag_sb[1], label="Im($\\rho_{\mathrm{S},12}$)")
    ax.plot(sym_sb_ref_time, sym_sb_ref_re_rho_12, 'o', color=colors_offdiag_sb[0], markevery=20, markersize=4)
    ax.plot(sym_sb_ref_time, sym_sb_ref_im_rho_12, 'o', color=colors_offdiag_sb[1], markevery=20, markersize=4)

    ax.set_title(f"SB (Sym): {label}", fontsize=12)
    ax.set_xlim(0, 20)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(x=4, color='black', linestyle='--')
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
        #ax.get_legend().remove()  # Remove legend from all other rows
    # Adjust shared x-axis labels
    if row_idx == 3:
        ax.set_xlabel('Time (1/Δ)', fontsize=12)  # SB model (first column)

# Plot SB model (First Column)
for row_idx, (file, label) in enumerate(asym_sb_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Diagonal elements
    re_rho_11 = np.real(rho[:, 0, 0])
    re_rho_22 = np.real(rho[:, 1, 1])
    asym_sb_ref_re_rho_11 = np.real(asym_sb_ref[:, 1])
    asym_sb_ref_re_rho_22 = np.real(asym_sb_ref[:, 4])

    # Off-diagonal elements
    re_rho_12 = np.real(rho[:, 0, 1])
    im_rho_12 = np.imag(rho[:, 0, 1])
    asym_sb_ref_re_rho_12 = np.real(asym_sb_ref[:, 2])
    asym_sb_ref_im_rho_12 = np.imag(asym_sb_ref[:, 2])

    ax = axes[row_idx, 1]
    ax.plot(time_pred, re_rho_11, color=colors_diag[0], label="$\\rho_{\mathrm{S},11}$")
    ax.plot(time_pred, re_rho_22, color=colors_diag[1], label="$\\rho_{\mathrm{S},22}$")
    ax.plot(asym_sb_ref_time, asym_sb_ref_re_rho_11, 'o', color=colors_diag[0], markevery=20, markersize=4)
    ax.plot(asym_sb_ref_time, asym_sb_ref_re_rho_22, 'o', color=colors_diag[1], markevery=20, markersize=4)

    ax.plot(time_pred, re_rho_12, color=colors_offdiag_sb[0], label="Re($\\rho_{\mathrm{S},12}$)")
    ax.plot(time_pred, im_rho_12, color=colors_offdiag_sb[1], label="Im($\\rho_{\mathrm{S},12}$)")
    ax.plot(asym_sb_ref_time, asym_sb_ref_re_rho_12, 'o', color=colors_offdiag_sb[0], markevery=20, markersize=4)
    ax.plot(asym_sb_ref_time, asym_sb_ref_im_rho_12, 'o', color=colors_offdiag_sb[1], markevery=20, markersize=4)

    ax.set_title(f"SB (Asym): {label}", fontsize=12)
    ax.set_xlim(0, 20)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axvline(x=4, color='black', linestyle='--')
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
        #ax.get_legend().remove()  # Remove legend from all other rows
    # Adjust shared x-axis labels
    if row_idx == 3:
        ax.set_xlabel('Time (1/Δ)', fontsize=12)  # SB model (first column)



#axes[3, 2].set_xlabel('Time (ps)', fontsize=12)


axes[0, 0].text(-0.20, 1.20, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.20, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.20, 1.20, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 0].text(-0.20, 1.20, 'D', transform=axes[3, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.20, 1.20, 'E', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.20, 'F', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.20, 1.20, 'G', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 1].text(-0.20, 1.20, 'H', transform=axes[3, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')


# Save and show the plot
plt.savefig('sb_dyn.pdf', format='pdf', dpi=300)
plt.show()

