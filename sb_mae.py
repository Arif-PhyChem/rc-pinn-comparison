import numpy as np

def calculate_mae(time_pred, rho_pred, ref_rho, diag_indices, offdiag_indices):
    """
    Calculate MAE for diagonal and off-diagonal elements.
    
    Parameters:
        time_pred: np.array
            Predicted time points.
        rho_pred: np.array
            Predicted density matrix elements.
        ref_rho: np.array
            Reference density matrix elements.
        diag_indices: list
            Indices for diagonal elements.
        offdiag_indices: list
            Indices for off-diagonal elements.
            
    Returns:
        diag_mae: np.array
            Mean absolute error for diagonal elements.
        offdiag_mae: np.array
            Mean absolute error for real and imaginary parts of off-diagonal elements.
    """
    diag_mae = np.zeros(len(diag_indices), dtype=float)
    offdiag_mae = np.zeros(2, dtype=float)  # Real and imaginary parts

    for i in range(len(time_pred)):
        flat_rho_pred = rho_pred[i, :, :].flatten()
        
        # Calculate MAE for diagonal elements
        for idx, j in enumerate(diag_indices):
            diag_mae[idx] += abs(ref_rho[i, j] - flat_rho_pred[j])
        
        # Calculate MAE for off-diagonal elements (real and imaginary parts)
        for j in offdiag_indices:
            offdiag_mae[0] += abs(np.real(ref_rho[i, j]) - np.real(flat_rho_pred[j]))
            offdiag_mae[1] += abs(np.imag(ref_rho[i, j]) - np.imag(flat_rho_pred[j]))

    diag_mae /= len(time_pred)
    offdiag_mae /= (len(time_pred) * len(offdiag_indices))
    return np.mean(diag_mae), offdiag_mae

# File paths and labels
sym_sb_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/sr_traj.npz', 'SR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/psr_traj.npz', 'PSR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/mr_traj.npz', 'MR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/sym_data/pmr_traj.npz', 'PMR-PINN')
]

# File paths and labels
asym_sb_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/sr_traj.npz', 'SR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/psr_traj.npz', 'PSR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/mr_traj.npz', 'MR-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/sb/asym_data/pmr_traj.npz', 'PMR-PINN')                                 ] 



# Reference data
sym_sb_ref = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-0.0_Delta-1.0_lambda-0.6_gamma-3.0_beta-1.0.npy')
sym_sb_ref_rho = sym_sb_ref[:, 1:]  # Exclude time column



# Reference data
asym_sb_ref = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-1.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy')
asym_sb_ref_rho = asym_sb_ref[:, 1:]  # Exclude time column


# Diagonal and off-diagonal indices
sb_diag_indices = [0, 3]
sb_offdiag_indices = [1]

# Calculate and print MAE for SB model
print("Sym SB Model MAE:")
for file, label in sym_sb_density_files:
    data = np.load(file)
    time_pred = data['time']
    rho_pred = data['rho']
    diag_mae, offdiag_mae = calculate_mae(time_pred, rho_pred, sym_sb_ref_rho, sb_diag_indices, sb_offdiag_indices)
    print(f"{label}: Diagonal MAE = {diag_mae:.6f}, Off-diagonal MAE (Real, Imag) = {offdiag_mae}")

# Calculate and print MAE for SB model
print("Asym SB Model MAE:")
for file, label in asym_sb_density_files:
    data = np.load(file)
    time_pred = data['time']
    rho_pred = data['rho']
    diag_mae, offdiag_mae = calculate_mae(time_pred, rho_pred, asym_sb_ref_rho, sb_diag_indices, sb_offdiag_indices)
    print(f"{label}: Diagonal MAE = {diag_mae:.6f}, Off-diagonal MAE (Real, Imag) = {offdiag_mae}")

