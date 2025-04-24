import prep_input

Xin = 'fmo_x_sr-pinn'
Yin = 'fmo_y_sr-pinn'
systemType = 'FMO'
n_states = 7
xlength = 41
time = 1
time_step = 0.005
ostl_steps = 80
dataPath = '/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/training_data/init_1'
prior = 0

prep_input.RCDYN_single_step(Xin ,
    Yin,
    systemType,
    n_states ,
    xlength ,
    time ,
    time_step ,
    dataPath ,
    prior )


Xin = 'fmo_x_psr-pinn'
Yin = 'fmo_y_psr-pinn'

prep_input.RCDYN_param_single_step(Xin ,
    Yin,
    systemType,
    n_states ,
    xlength ,
    time ,
    time_step ,
    dataPath ,
    prior )


Xin = 'fmo_x_mr-pinn'
Yin = 'fmo_y_mr-pinn'


prep_input.RCDYN_multi_steps(Xin ,
    Yin,
    systemType,
    n_states ,
    xlength ,
    time ,
    time_step ,
    ostl_steps,
    dataPath ,
    prior )


Xin = 'fmo_x_pmr-pinn'
Yin = 'fmo_y_pmr-pinn'

prep_input.RCDYN_param_multi_steps(Xin ,
    Yin,
    systemType,
    n_states ,
    xlength ,
    time ,
    time_step ,
    ostl_steps,
    dataPath ,
    prior )


