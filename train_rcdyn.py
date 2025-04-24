import re
import numpy as np
import ml_models





sb_x = ['sb_x_sr-pinn.npy', 'sb_x_psr-pinn.npy', 'sb_x_mr-pinn.npy', 'sb_x_pmr-pinn.npy']
sb_y = ['sb_y_sr-pinn.npy', 'sb_y_psr-pinn.npy', 'sb_y_mr-pinn.npy', 'sb_y_pmr-pinn.npy']


fmo_x = ['fmo_x_sr-pinn.npy', 'fmo_x_psr-pinn.npy', 'fmo_x_mr-pinn.npy', 'fmo_x_pmr-pinn.npy']
fmo_y = ['fmo_y_sr-pinn.npy', 'fmo_y_psr-pinn.npy', 'fmo_y_mr-pinn.npy', 'fmo_y_pmr-pinn.npy']





def training_step(x: str, 
                  y: str, 
                  n_states: int,
                  directory: str):
    
    
    #hyperopt_optim.optimize(x, y, n_states, ml_model, 100, 30, pinn)
    
    x_load = np.load(x)
    y_load = np.load(y)

    ml_models.CNN_LSTM_train(x_load, y_load, n_states, 3000, 500, directory)
    


# Train for SB

for (x, y) in zip(sb_x, sb_y):
    
    n_states = 2
    
    tmp_1 = re.split(r'_', x)[2]

    directory = 'sb_' + re.split(r'.npy', tmp_1)[0] # Directory where the trained models will be saved

    training_step(x, y, n_states, directory)


# Train for FMO

for (x, y) in zip(fmo_x, fmo_y):
    
    n_states = 2
    
    tmp_1 = re.split(r'_', x)[2]

    directory = 'fmo_' + re.split(r'.npy', tmp_1)[0] # Directory where the trained models will be saved

    training_step(x, y, n_states, directory)
