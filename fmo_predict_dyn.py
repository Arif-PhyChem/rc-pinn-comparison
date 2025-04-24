import run_dyn 

xx = np.load('/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/test_data/init_1/7_initial-1_gamma-400.0_lambda-40.0_temp-90.0.npy')

Xin = xx[0:41,1:]

model_path_sr = "FMO_sr-pinn_trained_models_model-54-tloss-1.899e-06-vloss-2.473e-07.keras"
model_path_psr = "FMO_psr-pinn_model-52-tloss-1.662e-06-vloss-2.566e-07.keras"
model_path_mr = "FMO_mr-pinn_model-1015-tloss-1.677e-06-vloss-1.871e-06.keras"
model_path_pmr = "FMO_pmr-pinn_model-958-tloss-1.857e-06-vloss-1.989e-06.keras"

n_states = 7
time = 1 # ps
time_step = 0.005
gamma = 400.0
lamb = 40.0
temp = 90.0 
ostl_steps = 80 # Time steps to be predicted in one shot (MR and PMR-PINNs)

traj_output_file = "FMO_sr-pinn_traj"
rc_single_step(Xin, 
              model_path_sr,  
              n_states, 
              time, 
              time_step, 
              traj_output_file)



traj_output_file = "FMO_psr-pinn_traj"

rc_param_single_step(Xin, 
              model_path_psr,  
              n_states, 
              time, 
              time_step, 
              traj_output_file,
                gamma, 
                lamb, 
                temp)
#
traj_output_file = "FMO_mt-pinn_traj"
rc_multi_steps(Xin, 
              model_path_mr,  
              n_states, 
              time, 
              time_step,
              ostl_steps,
             traj_output_file)




traj_output_file = "FMO_pmt-pinn_traj"
rc_param_multi_steps(Xin, 
              model_path_pmr,  
              n_states,
              time, 
              time_step,
              ostl_steps, 
              traj_output_file,
                gamma, 
                lamb, 
                temp)
