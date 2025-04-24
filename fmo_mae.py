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

fmo_1_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/st_traj.npz', 'ST-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/pst_traj.npz', 'PST-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/mt_traj.npz', 'MT-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_1/pmt_traj.npz', 'PMT-PINN')
]

fmo_6_density_files = [
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_6/st_traj.npz', 'ST-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_6/pst_traj.npz', 'PST-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_6/mt_traj.npz', 'MT-PINN'),
    ('/home/dell/arif/rcdyn_compare/1d_cnn_comparison/fmo/init_6/pmt_traj.npz', 'PMT-PINN')
]
# Reference data

fmo_1_ref = np.load('/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/test_data/init_1/7_initial-1_gamma-400.0_lambda-40.0_temp-90.0.npy')
fmo_1_ref_rho = fmo_1_ref[:, 1:]  # Exclude time column

fmo_6_ref = np.load('/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/test_data/init_6/7_initial-6_gamma-500.0_lambda-70.0_temp-450.0.npy')
fmo_6_ref_rho = fmo_6_ref[:, 1:]  # Exclude time column


fmo_diag_indices = [0, 8, 16, 24, 32, 40, 48]
fmo_offdiag_indices = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 17, 18, 19, 20, 25, 26, 27, 33, 34, 41]

#Calculate and print MAE for FMO complex
print("\nFMO Complex (site-1) MAE:")
for file, label in fmo_1_density_files:
    data = np.load(file)
    time_pred = data['time']
    rho_pred = data['rho']
    diag_mae, offdiag_mae = calculate_mae(time_pred, rho_pred, fmo_1_ref_rho, fmo_diag_indices, fmo_offdiag_indices)
    print(f"{label}: Diagonal MAE = {diag_mae:.6f}, Off-diagonal MAE (Real, Imag) = {offdiag_mae}")

#Calculate and print MAE for FMO complex
print("\nFMO Complex (site-6) MAE:")
for file, label in fmo_6_density_files:
    data = np.load(file)
    time_pred = data['time']
    rho_pred = data['rho']
    diag_mae, offdiag_mae = calculate_mae(time_pred, rho_pred, fmo_6_ref_rho, fmo_diag_indices, fmo_offdiag_indices)
    print(f"{label}: Diagonal MAE = {diag_mae:.6f}, Off-diagonal MAE (Real, Imag) = {offdiag_mae}")
