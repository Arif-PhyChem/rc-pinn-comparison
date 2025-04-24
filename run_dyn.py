import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import custom_loss as ml_loss


@register_keras_serializable(package='my_models')
def cnn_lstm_loss(y_true, y_pred):

    return ml_loss.pinn_loss(y_true, y_pred, n_states, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0)




def rc_single_step(Xin: np.ndarray, 
                   model_path: str,  
                   n_states: int, 
                   time: float, 
                   time_step: float, 
                   traj_output_file: str):

    model = tf.keras.models.load_model(model_path, custom_objects={'cnn_lstm_loss': cnn_lstm_loss})
    #Show the model architecture
    model.summary()
    tm = Xin.shape[0]
    time_range=0
    tt = time_range
    
    x = np.zeros((1, tm*n_states**2), dtype=float)
    yy = np.zeros((Xin.shape[0], Xin.shape[1]), dtype=complex)
    yy[:,:] = Xin[:, 0:n_states**2] # excluding the 1st column of time
    
    
    for i in range(0, tm + int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    
    print(len(time_range))
    
    a = 0; b = n_states
    labels = []
    m = 0
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
            m += 1
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    m = 0
    k = 0
    for j in range(0, tm):
        q = 0
        for p in labels:
            if p%divider == 0:
                x[m, q+k] = yy[j,p].real
                q += 1
            else:
                x[m, q+k] = yy[j,p].real
                q += 1
                x[m, q+k] = yy[j,p].imag
                q += 1
        k += n_states**2
    print('ml_dyn.RCDYN: Running recursive dynamics with CNN model......')
    y = np.zeros((len(time_range), n_states, n_states), dtype=complex)
    
    # Reconstruct the density matrix for the current time step
    y1 = np.zeros((1, n_states**2), dtype=float)

    rho_matrix = np.zeros((n_states, n_states), dtype=complex)
    
    a = 0; b = n_states**2;
    for i in range(0, tm):
        y1[0,:] = x[0, a:b]
        a = a + n_states**2
        b = b + n_states**2
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = y1[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = y1[0, idx]
                imag_part = y1[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y[i] = rho_matrix
    
    for i in range(tm, len(time_range)):
        x_pred = x
        x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
        yhat = model.predict(x_pred, verbose=0)
        x = np.append(x,yhat)
        x = x[n_states**2:]
        x = x.reshape(1, -1)
        # Reconstruct the density matrix for the current time step
    
        # Fill the diagonal elementS
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = yhat[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = yhat[0, idx]
                imag_part = yhat[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y[i] = rho_matrix

    trace = np.trace(y[:,:,:], axis1=1, axis2=2)
    np.savez('trace_'+traj_output_file, time=time_range, trace=np.real(trace))
    print('ml_dyn.OSTL: Trace is saved in a file  "' + "trace_"+traj_output_file + '"')
    eig, vec = np.linalg.eig(y[:,:, :])
    np.savez('eig_'+traj_output_file, time=time_range, eig=np.real(eig))
    print('ml_dyn.coeff_OSTL: Eigen values is saved in a file  "' + "eig_"+traj_output_file + '"')   
    np.savez(traj_output_file, time=time_range, rho=y)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')

def rc_multi_steps(Xin: np.ndarray, 
                   model_path: str,  
                   n_states: int, 
                   time: float, 
                   time_step: float, 
                   ostl_steps: int, 
                   traj_output_file: str):

    model = tf.keras.models.load_model(model_path, custom_objects={'cnn_lstm_loss': cnn_lstm_loss})
    #Show the model architecture
    model.summary()
    tm = Xin.shape[0]
    time_range=0
    tt = time_range
    
    x = np.zeros((1, tm*n_states**2), dtype=float)
    yy = np.zeros((Xin.shape[0], Xin.shape[1]), dtype=complex)
    yy[:,:] = Xin[:, 0:n_states**2] # excluding the 1st column of time
    
    
    for i in range(0, tm + int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    
    print(len(time_range))
    
    a = 0; b = n_states
    labels = []
    m = 0
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
            m += 1
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    m = 0
    k = 0
    for j in range(0, tm):
        q = 0
        for p in labels:
            if p%divider == 0:
                x[m, q+k] = yy[j,p].real
                q += 1
            else:
                x[m, q+k] = yy[j,p].real
                q += 1
                x[m, q+k] = yy[j,p].imag
                q += 1
        k += n_states**2
    print('ml_dyn.RCDYN: Running recursive dynamics with CNN model......')
    y = np.zeros((len(time_range), n_states, n_states), dtype=complex)
    
    # Reconstruct the density matrix for the current time step
    y1 = np.zeros((1, n_states**2), dtype=float)

    rho_matrix = np.zeros((n_states, n_states), dtype=complex)
    
    a = 0; b = n_states**2;
    for i in range(0, tm):
        y1[0,:] = x[0, a:b]
        a = a + n_states**2
        b = b + n_states**2
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = y1[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = y1[0, idx]
                imag_part = y1[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y[i] = rho_matrix
    zz = tm
    for i in range(tm, len(time_range), ostl_steps):
        x_pred = x
        x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
        yhat = model.predict(x_pred, verbose=0)
        print(yhat.shape)
        x = np.append(x,yhat)
        x = x[ostl_steps*n_states**2:]
        x = x.reshape(1, -1)
        # Reconstruct the density matrix for the current time step
    
        # Fill the diagonal elementS
        a = 0; b = n_states**2;
        for kk in range(0, ostl_steps):
            y1 = yhat[0, a:b]
            idx = 0
            for j in range(n_states):
                if j > 0:  #
                    idx += 2 * (n_states - j) + 1

                rho_matrix[j, j] = y1[idx]  

            # Fill the off-diagonal elements (real + imaginary)
            idx = 1  # Start after the diagonal elements
            for row in range(0, n_states-1):
                for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                    real_part = y1[idx]
                    imag_part = y1[idx + 1]
                    rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                    rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                    if col == n_states -1:
                        idx += 3        # skip diagonal terms
                    else:
                        idx += 2  # Move to the next real/imag pair
            
            # Store the reconstructed matrix in the 3D array

            y[zz] = rho_matrix
            zz += 1
            a = a + n_states**2
            b = b + n_states**2

    trace = np.trace(y[:,:,:], axis1=1, axis2=2)
    np.savez('trace_'+traj_output_file, time=time_range, trace=np.real(trace))
    print('ml_dyn.OSTL: Trace is saved in a file  "' + "trace_"+traj_output_file + '"')
    eig, vec = np.linalg.eig(y[:,:, :])
    np.savez('eig_'+traj_output_file, time=time_range, eig=np.real(eig))
    print('ml_dyn.coeff_OSTL: Eigen values is saved in a file  "' + "eig_"+traj_output_file + '"')   
    np.savez(traj_output_file, time=time_range, rho=y)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')


def rc_param_single_step(Xin: np.ndarray, 
                   model_path: str,  
                   n_states: int, 
                   time: float, 
                   time_step: float, 
                   traj_output_file: str, 
                    gamma: float, 
                         lamb: float, 
                         beta: float):

    model = tf.keras.models.load_model(model_path, custom_objects={'cnn_lstm_loss': cnn_lstm_loss})
    #Show the model architecture
    model.summary()
    tm = Xin.shape[0]
    time_range=0
    tt = time_range
    nsp = 3

    x = np.zeros((1, nsp+tm*n_states**2), dtype=float)
    yy = np.zeros((Xin.shape[0], Xin.shape[1]), dtype=complex)
    yy[:,:] = Xin[:, 0:n_states**2] # excluding the 1st column of time
    
    
    for i in range(0, tm + int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    
    print(len(time_range))
    
    a = 0; b = n_states
    labels = []
    m = 0
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
            m += 1
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    

    x[0, 0] = gamma/10.0 # Normalizing 
    x[0, 1] = lamb/1.0  # Normalizing 
    x[0, 2] = beta/1.0  # Normalizing 

    m = 0
    k = 0
    for j in range(0, tm):
        q = 0
        for p in labels:
            if p%divider == 0:
                x[m, nsp+q+k] = yy[j,p].real
                q += 1
            else:
                x[m, nsp+q+k] = yy[j,p].real
                q += 1
                x[m, nsp+q+k] = yy[j,p].imag
                q += 1
        k += n_states**2
    print('ml_dyn.RCDYN: Running recursive dynamics with CNN model......')
    y = np.zeros((len(time_range), n_states, n_states), dtype=complex)
    
    # Reconstruct the density matrix for the current time step
    y1 = np.zeros((1, n_states**2), dtype=float)

    rho_matrix = np.zeros((n_states, n_states), dtype=complex)
    
    a = 0+nsp; b = nsp+n_states**2;
    for i in range(0, tm):
        y1[0,:] = x[0, a:b]
        a = a + n_states**2
        b = b + n_states**2
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = y1[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = y1[0, idx]
                imag_part = y1[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y[i] = rho_matrix
    
    for i in range(tm, len(time_range)):
        x_pred = x
        x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
        yhat = model.predict(x_pred, verbose=0)
        x = np.append(x,yhat)
        x1 = x[nsp+n_states**2:]
        x = x[0:nsp]
        x = np.append(x, x1)
        x = x.reshape(1, -1)
        # Reconstruct the density matrix for the current time step
    
        # Fill the diagonal elementS
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = yhat[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = yhat[0, idx]
                imag_part = yhat[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y[i] = rho_matrix

    trace = np.trace(y[:,:,:], axis1=1, axis2=2)
    np.savez('trace_'+traj_output_file, time=time_range, trace=np.real(trace))
    print('ml_dyn.OSTL: Trace is saved in a file  "' + "trace_"+traj_output_file + '"')
    eig, vec = np.linalg.eig(y[:,:, :])
    np.savez('eig_'+traj_output_file, time=time_range, eig=np.real(eig))
    print('ml_dyn.coeff_OSTL: Eigen values is saved in a file  "' + "eig_"+traj_output_file + '"')   
    np.savez(traj_output_file, time=time_range, rho=y)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')

def rc_param_multi_steps(Xin: np.ndarray, 
                   model_path: str,  
                   n_states: int, 
                   time: float, 
                   time_step: float, 
                    ostl_steps: int, 
                   traj_output_file: str, 
                    gamma: float, 
                         lamb: float, 
                         beta: float):

    model = tf.keras.models.load_model(model_path, custom_objects={'cnn_lstm_loss': cnn_lstm_loss})
    #Show the model architecture
    model.summary()
    tm = Xin.shape[0]
    time_range=0
    tt = time_range
    nsp = 3

    x = np.zeros((1, nsp+tm*n_states**2), dtype=float)
    yy = np.zeros((Xin.shape[0], Xin.shape[1]), dtype=complex)
    yy[:,:] = Xin[:, 0:n_states**2] # excluding the 1st column of time
    
    
    for i in range(0, tm + int(time/time_step)-1):
        tt += time_step
        time_range = np.append(time_range, tt)
    
    print(len(time_range))
    
    a = 0; b = n_states
    labels = []
    m = 0
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
            m += 1
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    

    x[0, 0] = gamma/10.0 # Normalizing 
    x[0, 1] = lamb/1.0  # Normalizing 
    x[0, 2] = beta/1.0  # Normalizing 

    m = 0
    k = 0
    for j in range(0, tm):
        q = 0
        for p in labels:
            if p%divider == 0:
                x[m, nsp+q+k] = yy[j,p].real
                q += 1
            else:
                x[m, nsp+q+k] = yy[j,p].real
                q += 1
                x[m, nsp+q+k] = yy[j,p].imag
                q += 1
        k += n_states**2
    print('ml_dyn.RCDYN: Running recursive dynamics with CNN model......')
    y = np.zeros((len(time_range), n_states, n_states), dtype=complex)
    
    # Reconstruct the density matrix for the current time step
    y1 = np.zeros((1, n_states**2), dtype=float)

    rho_matrix = np.zeros((n_states, n_states), dtype=complex)
    
    a = 0+nsp; b = nsp+n_states**2;
    for i in range(0, tm):
        y1[0,:] = x[0, a:b]
        a = a + n_states**2
        b = b + n_states**2
        idx = 0
        for j in range(n_states):
            if j > 0:  #
                idx += 2 * (n_states - j) + 1

            rho_matrix[j, j] = y1[0, idx]  

        # Fill the off-diagonal elements (real + imaginary)
        idx = 1  # Start after the diagonal elements
        for row in range(0, n_states-1):
            for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                real_part = y1[0, idx]
                imag_part = y1[0, idx + 1]
                rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                if col == n_states -1:
                    idx += 3        # skip diagonal terms
                else:
                    idx += 2  # Move to the next real/imag pair
        
        # Store the reconstructed matrix in the 3D array
        
        y[i] = rho_matrix
        
    zz = tm
    for i in range(tm, len(time_range), ostl_steps):
        x_pred = x
        x_pred = x_pred.reshape(1, x.shape[1],1) # reshape the input 
        yhat = model.predict(x_pred, verbose=0)
        x = np.append(x,yhat)
        x1 = x[nsp+ostl_steps*n_states**2:]
        x = x[0:nsp]
        x = np.append(x, x1)
        x = x.reshape(1, -1)

        # Fill the diagonal elementS
        a = 0; b = n_states**2;
        for kk in range(0, ostl_steps):
            y1 = yhat[0, a:b]
            idx = 0
            for j in range(n_states):
                if j > 0:  #
                    idx += 2 * (n_states - j) + 1

                rho_matrix[j, j] = y1[idx]  

            # Fill the off-diagonal elements (real + imaginary)
            idx = 1  # Start after the diagonal elements
            for row in range(0, n_states-1):
                for col in range(row + 1, n_states):  # Only fill upper triangle (lower will be conjugate)
                    real_part = y1[idx]
                    imag_part = y1[idx + 1]
                    rho_matrix[row, col] = real_part + 1j * imag_part  # rho_ij
                    rho_matrix[col, row] = np.conj(rho_matrix[row, col])  # rho_ji is complex conjugate of rho_ij
                    if col == n_states -1:
                        idx += 3        # skip diagonal terms
                    else:
                        idx += 2  # Move to the next real/imag pair
            
            # Store the reconstructed matrix in the 3D array

            y[zz] = rho_matrix
            zz += 1
            a = a + n_states**2
            b = b + n_states**2
   
    trace = np.trace(y[:,:,:], axis1=1, axis2=2)
    np.savez('trace_'+traj_output_file, time=time_range, trace=np.real(trace))
    print('ml_dyn.OSTL: Trace is saved in a file  "' + "trace_"+traj_output_file + '"')
    eig, vec = np.linalg.eig(y[:,:, :])
    np.savez('eig_'+traj_output_file, time=time_range, eig=np.real(eig))
    print('ml_dyn.coeff_OSTL: Eigen values is saved in a file  "' + "eig_"+traj_output_file + '"')   
    np.savez(traj_output_file, time=time_range, rho=y)
    print('ml_dyn.OSTL: Dynamics is saved in a file  "' + traj_output_file + '"')

