import run_dyn


xx_asym = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-1.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy')#
xx_sym = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-0.0_Delta-1.0_lambda-0.6_gamma-3.0_beta-1.0.npy')

Xin_asym = xx_asym[0:81,1:]
Xin_sym = xx_sym[0:81,1:]


model_path_sr = "SB_sr-pinn_model-11-tloss-3.947e-06-vloss-1.405e-06.keras"
model_path_psr = "SB_psr-pinn_model-11-tloss-4.545e-06-vloss-1.441e-06.keras"
model_path_mr = "SB_mr-pinn_model-440-tloss-4.573e-06-vloss-4.348e-06.keras"
model_path_pmr = "SB_pmr_pinn_model-403-tloss-4.156e-06-vloss-4.493e-06.keras"

n_states = 2
time = 16 # propagation time
time_step = 0.05
gamma = 9.0
lamb = 0.6
beta = 1.0 
ostl_steps = 40 # Time steps to be predicted in one shot (MR and PMR-PINNs)


# Prediction for Symmetric model

Xin = Xin_sym 

traj_output_file = "Sym_SB_sr-pinn_traj"
run_dyn.rc_single_step(Xin, 
              model_path_sr,  
              n_states, 
              time, 
              time_step, 
              traj_output_file)

traj_output_file = "Sym_SB_psr-pinn_traj"

run_dyn.rc_param_single_step(Xin, 
              model_path_psr,  
              n_states, 
              time, 
              time_step, 
              traj_output_file,
                gamma, 
                lamb, 
                beta)

traj_output_file = "Sym_SB_mt-pinn_traj"
run_dyn.rc_multi_steps(Xin, 
              model_path_mr,  
              n_states, 
              time, 
              time_step,
              ostl_steps,
             traj_output_file)




traj_output_file = "Sym_SB_pmr-pinn_traj"
run_dyn.rc_param_multi_steps(Xin, 
              model_path_pmr,  
              n_states,
              time, 
              time_step,
              ostl_steps, 
              traj_output_file,
                gamma, 
                lamb, 
                beta)


# Prediction for Asymmetric model

Xin = Xin_asym 

traj_output_file = "Asym_SB_sr-pinn_traj"
run_dyn.rc_single_step(Xin, 
              model_path_sr,  
              n_states, 
              time, 
              time_step, 
              traj_output_file)

traj_output_file = "Asym_SB_psr-pinn_traj"

run_dyn.rc_param_single_step(Xin, 
              model_path_psr,  
              n_states, 
              time, 
              time_step, 
              traj_output_file,
                gamma, 
                lamb, 
                beta)

traj_output_file = "Asym_SB_mt-pinn_traj"
run_dyn.rc_multi_steps(Xin, 
              model_path_mr,  
              n_states, 
              time, 
              time_step,
              ostl_steps,
             traj_output_file)




traj_output_file = "Asym_SB_pmr-pinn_traj"
run_dyn.rc_param_multi_steps(Xin, 
              model_path_pmr,  
              n_states,
              time, 
              time_step,
              ostl_steps, 
              traj_output_file,
                gamma, 
                lamb, 
                beta)
