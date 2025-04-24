import prep_input

Xin = 'sb_x_sr-pinn'
Yin = 'sb_y_sr-pinn'
systemType = 'SB'
n_states = 2
xlength = 81
time = 20
time_step = 0.05
ostl_steps = 40
dataPath = '/home/dell/arif/pypackage/sb/data/training_data/combined'
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
#

Xin = 'sb_x_psr-pinn'
Yin = 'sb_y_psr-pinn'
#
prep_input.RCDYN_param_single_step(Xin ,
    Yin,
    systemType,
    n_states ,
    xlength ,
    time ,
    time_step ,
    dataPath ,
    prior )
#

Xin = 'sb_x_mr-pinn'
Yin = 'sb_y_mr-pinn'


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


Xin = 'sb_x_pmr-pinn'
Yin = 'sb_y_pmr-pinn'

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


