# rc-pinn-comparison
Code and data for our recent paper **From short-sighted to far-sighted: A comparative study of recursive machine learning approaches for open quantum systems** (https://arxiv.org/abs/2504.02218)

1. **fmo_data folder:** Training and test data for FMO complex. We are just providing the file names. You can download QD3SET-1 database (https://doi.org/10.3389/fphy.2023.1223973) to access the real data 


2. **sb_data folder :** Training and test data for SB model. We are just providing the file names. You can download QD3SET-1 database (https://doi.org/10.3389/fphy.2023.1223973) to access the real data 


3. **FMO trained models (Only for site-1):** 

    1. FMO_mr-pinn_model-1015-tloss-1.677e-06-vloss-1.871e-06.keras
    2. FMO_pmr-pinn_model-958-tloss-1.857e-06-vloss-1.989e-06.keras
    3. FMO_psr-pinn_model-52-tloss-1.662e-06-vloss-2.566e-07.keras
    4. FMO_sr-pinn_model-54-tloss-1.899e-06-vloss-2.473e-07.keras

4. **SB trained models:**
    
    1. SB_mt-pinn_model-440-tloss-4.573e-06-vloss-4.348e-06.keras
    2. SB_pmt-pinn_model-403-tloss-4.156e-06-vloss-4.493e-06.keras
    3. SB_pst-pinn_model-11-tloss-4.545e-06-vloss-1.441e-06.keras
    4. SB_st-pinn_model-11-tloss-3.947e-06-vloss-1.405e-06.keras


5. **custom_loss.py:** Custom loss function including all loss terms

6. **fmo_mae.py:** Calculating the average mean absolute error (MAE) for FMO complex


7. **fmo_predict_dyn.py:** Predicting the dynamics for the test trajectory (FMO compelx)


8. **fmo_prep_data.py:** Preparing training files (X & Y) for FMO complex


9. **ml_models.py:** A module with CNN-LSTM architecture


10. **plot_fmo_dyn.py:** Plot dynamics for FMO complex


11. **plot_sb_dynamics.py** Plot SB dynamics


12. **prep_input.py:** A module with code for training data preparation 

13. **sb_mae.py:** Calculating the average mean absolute error (MAE) for SB model

14. **sb_predict_dyn.py:** Predicting the dynamics for the test trajectory (SB model)

15. **sb_prep_data.py:** Preparing training files (X & Y) for SB model

16. **train_rcdyn.py:**  Training CNN-LSTM models for both SB model and FMO complex
