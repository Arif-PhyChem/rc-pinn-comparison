import glob 
import os 
import re
import h5py
import numpy as np


def RCDYN_single_step(Xin: str,
        Yin: str,
        systemType: int, 
        n_states: int,
        xlength: int,
        time: float,
        time_step: float,
        dataPath: str,
        prior: float):

    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    xx = all_files[file_name]
    tot_length = xx[0:int(round((time+time_step)/time_step) + xlength),:].shape[0]
    file_count = j
    print('prep_input.RCDYN: Number of trajectories =', file_count)
    m = (tot_length - xlength) * file_count
    print(m) 
    labels = []
    a = 0; b = n_states
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
   
    x = np.zeros((int(m), xlength*n_states**2), dtype=float)
    y = np.zeros((int(m), n_states**2), dtype=float)
    yy = np.zeros((xlength+1, n_states**2), dtype=complex)
    m = 0
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        for i in range(0, (tot_length - xlength)):
            yy[:,:] = df[i:xlength+i+1, 1:n_states**2+1] # excluding the 1st column of time
            k = 0
            for j in range(0, xlength):
                q = 0
                for p in labels:
                    if p%divider == 0:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                    else:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                        x[m,q+k] = yy[j,p].imag
                        q += 1
                k += n_states**2
            q = 0
            for p in labels:
                if p%divider == 0:
                    y[m,q] = yy[j+1,p].real
                    q += 1
                else:
                    y[m,q] = yy[j+1,p].real
                    q += 1
                    y[m,q] = yy[j+1,p].imag
                    q += 1
            m += 1
    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y - prior) # the target values are saved as Yin

def RCDYN_param_single_step(Xin: str,
        Yin: str,
        systemType: str,
        n_states: int,
        xlength: int,
        time: float,
        time_step: float,
        dataPath: str,
        prior: float):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    
    if systemType == 'SB':
        lambNorm = 1.0
        gammaNorm = 10.0
        tempNorm = 1.0
    if systemType == 'FMO':
        lambNorm = 530.0
        gammaNorm = 500.0
        tempNorm = 510.0

    file_count = j
    
    print('prep_input.OSTL: Normalizing gamma, lambda and temperature using the following',
            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)

    xx = all_files[file_name]
    tot_length = xx[0:int(round((time+time_step)/time_step) + xlength),:].shape[0]
    file_count = j
    print('prep_input.RCDYN: Number of trajectories =', file_count)
    m = (tot_length - xlength) * file_count
    print(m)
    nsp = 3 # number of simulation parameters
    labels = []
    a = 0; b = n_states
    
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    
        
    x = np.zeros((int(m), nsp + xlength*n_states**2), dtype=float)
    y = np.zeros((int(m), n_states**2), dtype=float)
    yy = np.zeros((xlength+1, n_states**2), dtype=complex)
    
    m = 0

    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        if systemType == 'SB': 
            pr = re.split(r'-', file_name)
            pp = re.split(r'_', pr[1])
            epsilon = pp[0]
            pp = re.split(r'_', pr[2])
            Delta = pp[0]
            pp = re.split(r'_', pr[3])
            lamb = float(pp[0])/lambNorm
            pp = re.split(r'_', pr[4])
            gamma = float(pp[0])/gammaNorm
            pp = re.split(r'.n', pr[5])
            temp = float(pp[0])/tempNorm
        else:
            pr = re.split(r'_', file_name)
            pp = re.split(r'-', pr[2]) # extracting value of gamma
            gamma = float(pp[1])/gammaNorm
            pp = re.split(r'-', pr[3]) # extract value of lambda 
            lamb = float(pp[1])/lambNorm
            pp = re.split(r'-', pr[4])
            pr = re.split(r'.npy', pp[1]) # extract value of temperature
            temp = float(pr[0])/tempNorm
        for i in range(0, tot_length - xlength):
            yy = df[i:xlength+i+1, 1:n_states**2+1] # excluding the 1st column of time
            k = 0
            for j in range(0, xlength):
                q = nsp
                x[m, 0] = gamma
                x[m, 1] = lamb
                x[m, 2] = temp
                for p in labels:
                    if p%divider == 0:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                    else:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                        x[m,q+k] = yy[j,p].imag
                        q += 1
                k += n_states**2
            q = 0
            for p in labels:
                if p%divider == 0:
                    y[m,q] = yy[j+1,p].real
                    q += 1
                else:
                    y[m,q] = yy[j+1,p].real
                    q += 1
                    y[m,q] = yy[j+1,p].imag
                    q += 1
            m += 1
    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y - prior) # the target values are saved as Yin

def RCDYN_multi_steps(Xin: str,
        Yin: str,
        systemType: str,
        n_states: int,
        xlength: int,
        time: float,
        time_step: float,
        ostl_steps: int,
        dataPath: str,
        prior: float):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    xx = all_files[file_name]
    tot_length = xx[0:int(round((time+time_step)/time_step)),:].shape[0]
    file_count = j
    print('prep_input.RCDYN: Number of trajectories =', file_count)
    print(tot_length, xlength, ostl_steps) 
    print(np.arange(0, tot_length - xlength, ostl_steps))
    num = 0
    for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
        num += 1  
            
    m = num * file_count
    print(m)

    labels = []
    a = 0; b = n_states
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
        a += n_states + 1 
        b += n_states

    divider = n_states + 1
    
    x = np.zeros((int(m), xlength*n_states**2), dtype=float)
    y = np.zeros((int(m), n_states**2*ostl_steps), dtype=float)
    
    m = 0
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
            yy = df[i*ostl_steps:xlength + (i+1) * ostl_steps, 1:n_states**2+1] # excluding the 1st column of time
            k = 0
            for j in range(0, xlength):
                q = 0
                for p in labels:
                    if p%divider == 0:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                    else:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                        x[m,q+k] = yy[j,p].imag
                        q += 1
                k += n_states**2
            
            k = 0
            for l in range(j+1, yy.shape[0]):
                q = 0
                for p in labels:
                    if p%divider == 0:
                        y[m,q+k] = yy[l,p].real
                        q += 1
                    else:
                        y[m,q+k] = yy[l,p].real
                        q += 1
                        y[m,q+k] = yy[l,p].imag
                        q += 1
                k += n_states**2
            m += 1
    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y - prior) # the target values are saved as Yin

def RCDYN_param_multi_steps(Xin: str,
        Yin: str,
        systemType: str,
        n_states: int,
        xlength: int,
        time: float,
        time_step: float,
        ostl_steps: int,
        dataPath: str,
        prior: float):
    all_files = {}
    j = 0
    print('=================================================================')
    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        all_files[file_name] = np.load(files)
        j += 1
    
    if systemType == 'SB':
        lambNorm = 1.0
        gammaNorm = 10.0
        tempNorm = 1.0
    if systemType == 'FMO':
        lambNorm = 530.0
        gammaNorm = 500.0
        tempNorm = 510.0

    file_count = j
    
    print('prep_input.OSTL: Normalizing gamma, lambda and temperature using the following',
            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)

    xx = all_files[file_name]
    tot_length = xx[0:int(round((time+time_step)/time_step)),:].shape[0]
    file_count = j
    print('prep_input.RCDYN: Number of trajectories =', file_count)
    nsp = 3
    num = 0
    for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
        num += 1  
            
    m = num * file_count
    print(m)

    nsp = 3 # number of simulation parameters
    labels = []
    a = 0; b = n_states
    
    for i in range(0, n_states):
        for j in range(a, b):
            labels.append(j)
        a += n_states + 1 
        b += n_states
    divider = n_states + 1
    
    x = np.zeros((int(m), nsp + xlength*n_states**2), dtype=float)
    y = np.zeros((int(m), n_states**2*ostl_steps), dtype=float)
    
    m = 0

    for files in glob.glob(dataPath+'/*'):
        file_name = os.path.basename(files)
        df = all_files[file_name]
        if systemType == 'SB': 
            pr = re.split(r'-', file_name)
            pp = re.split(r'_', pr[1])
            epsilon = pp[0]
            pp = re.split(r'_', pr[2])
            Delta = pp[0]
            pp = re.split(r'_', pr[3])
            lamb = float(pp[0])/lambNorm
            pp = re.split(r'_', pr[4])
            gamma = float(pp[0])/gammaNorm
            pp = re.split(r'.n', pr[5])
            temp = float(pp[0])/tempNorm
        else:
            pr = re.split(r'_', file_name)
            pp = re.split(r'-', pr[2]) # extracting value of gamma
            gamma = float(pp[1])/gammaNorm
            pp = re.split(r'-', pr[3]) # extract value of lambda 
            lamb = float(pp[1])/lambNorm
            pp = re.split(r'-', pr[4])
            pr = re.split(r'.npy', pp[1]) # extract value of temperature
            temp = float(pr[0])/tempNorm

        for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
            yy = df[i*ostl_steps:xlength + (i+1) * ostl_steps, 1:n_states**2+1] # excluding the 1st column of time
            k = 0
            for j in range(0, xlength):
                q = nsp
                x[m, 0] = gamma
                x[m, 1] = lamb
                x[m, 2] = temp
                for p in labels:
                    if p%divider == 0:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                    else:
                        x[m,q+k] = yy[j,p].real
                        q += 1
                        x[m,q+k] = yy[j,p].imag
                        q += 1
                k += n_states**2
            
            k = 0
            for l in range(j+1, yy.shape[0]):
                q = 0
                for p in labels:
                    if p%divider == 0:
                        y[m,q+k] = yy[l,p].real
                        q += 1
                    else:
                        y[m,q+k] = yy[l,p].real
                        q += 1
                        y[m,q+k] = yy[l,p].imag
                        q += 1
                k += n_states**2
            m += 1

    np.save(Xin, x) # the input is saved as Xin
    np.save(Yin, y - prior) # the target values are saved as Yin

#def RCDYN_multi_steps(Xin: str,
#        Yin: str,
#        systemType: str,
#        n_states: int,
#        dataCol: int, 
#        xlength: int,
#        time: float,
#        time_step: float,
#        ostl_steps: int,
#        dataPath: str,
#        prior: float):
#    all_files = {}
#    j = 0
#    print('=================================================================')
#    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
#    for files in glob.glob(dataPath+'/*'):
#        file_name = os.path.basename(files)
#        all_files[file_name] = np.load(files)
#        j += 1
#    
#    if systemType == 'SB':
#        lambNorm = 1.0
#        gammaNorm = 10.0
#        tempNorm = 1.0
#    if systemType == 'FMO':
#        lambNorm = 530.0
#        gammaNorm = 500.0
#        tempNorm = 510.0
#
#    file_count = j
#    
#    print('prep_input.OSTL: Normalizing gamma, lambda and temperature using the following',
#            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)
#
#
#    xx = all_files[file_name]
#    tot_length = xx[0:int(round((time+time_step)/time_step) + xlength),:].shape[0]
#    file_count = j
#    print('prep_input.RCDYN: Number of trajectories =', file_count)
#    num = 0
#    for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
#        num += 1  
#    print(num) 
#    m = num * file_count
#    print(m)
#
#    nsp = 3 # number of simulation parameters
#    labels = []
#    a = 0; b = n_states
#    for i in range(0, n_states):
#        for j in range(a, b):
#            labels.append(j)
#        a += n_states + 1 
#        b += n_states
#    divider = n_states + 1
#    
#    xxx = np.zeros((int(m), nsp + xlength*n_states**2 + (num-1)*ostl_steps*n_states**2), dtype=float)
#    yyy = np.zeros((int(m), n_states**2*ostl_steps), dtype=float)
#    
#    m = 0
#
#    for files in glob.glob(dataPath+'/*'):
#        file_name = os.path.basename(files)
#        df = all_files[file_name]
#        if systemType == 'SB': 
#            x = re.split(r'-', file_name)
#            y = re.split(r'_', x[1])
#            epsilon = y[0]
#            y = re.split(r'_', x[2])
#            Delta = y[0]
#            y = re.split(r'_', x[3])
#            lamb = float(y[0])/lambNorm
#            y = re.split(r'_', x[4])
#            gamma = float(y[0])/gammaNorm
#            y = re.split(r'.n', x[5])
#            temp = float(y[0])/tempNorm
#        else:
#            x = re.split(r'_', file_name)
#            y = re.split(r'-', x[2]) # extracting value of gamma
#            gamma = float(y[1])/gammaNorm
#            y = re.split(r'-', x[3]) # extract value of lambda 
#            lamb = float(y[1])/lambNorm
#            y = re.split(r'-', x[4])
#            x = re.split(r'.npy', y[1]) # extract value of temperature
#            temp = float(x[0])/tempNorm
#        
#        if dataCol != None:
#            for i in range(0, (tot_length - xlength)):
#                yy = df[:, dataCol]
#                if (dataCol -1)%divider == 0:
#                    x[m,:] = yy[i:xlength+i].real
#                    y[m,0] = yy[i+xlength].real
#                else:
#                    x[m,0:xlength] = yy[i:xlength+i].real
#                    y[m,0] = yy[i+xlength].real
#                    x[m,xlength:] = yy[i:xlength+i].imag
#                    y[m,1] = yy[i+xlength].imag
#                m += 1
#        if dataCol == None:
#            for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
#                yy = df[0:xlength+(i+1)*ostl_steps, 1:n_states**2+1] # excluding the 1st column of time
#                k = 0
#                for j in range(0, yy.shape[0] - ostl_steps):
#                    q = nsp
#                    xxx[m, 0] = gamma
#                    xxx[m, 1] = lamb
#                    xxx[m, 2] = temp
#                    for p in labels:
#                        if p%divider == 0:
#                            xxx[m,q+k] = yy[j,p].real
#                            q += 1
#                        else:
#                            xxx[m,q+k] = yy[j,p].real
#                            q += 1
#                            xxx[m,q+k] = yy[j,p].imag
#                            q += 1
#                    k += n_states**2
#                k = 0
#                for l in range(j+1, yy.shape[0]):
#                    q = 0
#                    for p in labels:
#                        if p%divider == 0:
#                            yyy[m,q+k] = yy[l,p].real
#                            q += 1
#                        else:
#                            yyy[m,q+k] = yy[l,p].real
#                            q += 1
#                            yyy[m,q+k] = yy[l,p].imag
#                            q += 1
#                    k += n_states**2
#                m += 1
#    np.save(Xin, xxx) # the input is saved as Xin
#    np.save(Yin, yyy) # the target values are saved as Yin
#
#
#
#def RCDYN_multi_equal(Xin: str,
#        Yin: str,
#        systemType: str,
#        n_states: int,
#        dataCol: int, 
#        xlength: int,
#        time: float,
#        time_step: float,
#        ostl_steps: int,
#        dataPath: str,
#        prior: float):
#    all_files = {}
#    j = 0
#    print('=================================================================')
#    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
#    for files in glob.glob(dataPath+'/*'):
#        file_name = os.path.basename(files)
#        all_files[file_name] = np.load(files)
#        j += 1
#    
#    if systemType == 'SB':
#        lambNorm = 1.0
#        gammaNorm = 10.0
#        tempNorm = 1.0
#    if systemType == 'FMO':
#        lambNorm = 530.0
#        gammaNorm = 500.0
#        tempNorm = 510.0
#
#    file_count = j
#    
#    print('prep_input.OSTL: Normalizing gamma, lambda and temperature using the following',
#            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)
#
#    xx = all_files[file_name]
#    tot_length = xx[0:int(round((time+time_step)/time_step) + xlength),:].shape[0]
#    file_count = j
#    print('prep_input.RCDYN: Number of trajectories =', file_count)
#    m = (tot_length - xlength) * file_count
#    nsp = 3 # number of simulation parameters
#    labels = []
#    a = 0; b = n_states
#    for i in range(0, n_states):
#        for j in range(a, b):
#            labels.append(j)
#        a += n_states + 1 
#        b += n_states
#    divider = n_states + 1
#    Xin, Yin = [], []
#    for files in glob.glob(dataPath+'/*'):
#        file_name = os.path.basename(files)
#        df = all_files[file_name]
#        if systemType == 'SB': 
#            x = re.split(r'-', file_name)
#            y = re.split(r'_', x[1])
#            epsilon = y[0]
#            y = re.split(r'_', x[2])
#            Delta = y[0]
#            y = re.split(r'_', x[3])
#            lamb = float(y[0])/lambNorm
#            y = re.split(r'_', x[4])
#            gamma = float(y[0])/gammaNorm
#            y = re.split(r'.n', x[5])
#            temp = float(y[0])/tempNorm
#        else:
#            x = re.split(r'_', file_name)
#            y = re.split(r'-', x[2]) # extracting value of gamma
#            gamma = float(y[1])/gammaNorm
#            y = re.split(r'-', x[3]) # extract value of lambda 
#            lamb = float(y[1])/lambNorm
#            y = re.split(r'-', x[4])
#            x = re.split(r'.npy', y[1]) # extract value of temperature
#            temp = float(x[0])/tempNorm
#        if dataCol != None:
#            for i in range(0, (tot_length - xlength)):
#                yy = df[:, dataCol]
#                if (dataCol -1)%divider == 0:
#                    x[m,:] = yy[i:xlength+i].real
#                    y[m,0] = yy[i+xlength].real
#                else:
#                    x[m,0:xlength] = yy[i:xlength+i].real
#                    y[m,0] = yy[i+xlength].real
#                    x[m,xlength:] = yy[i:xlength+i].imag
#                    y[m,1] = yy[i+xlength].imag
#                m += 1
#        if dataCol == None:
#            for i in range(0, len(np.arange(0, tot_length - xlength, ostl_steps))):
#                yy = df[i*ostl_steps:xlength + (i+1)*ostl_steps, 1:n_states**2+1] # excluding the 1st column of time
#                x = np.zeros((1, xlength*n_states**2 + nsp), dtype=float) # nsp is the number of simulation parameters
#                y = np.zeros((1, ostl_steps*n_states**2), dtype=float)
#                k = 0
#                for j in range(0, yy.shape[0] - ostl_steps):
#                    q = nsp
#                    x[0, 0] = gamma
#                    x[0, 1] = lamb
#                    x[0, 2] = temp
#                    for p in labels:
#                        if p%divider == 0:
#                            x[0,q+k] = yy[j,p].real
#                            q += 1
#                        else:
#                            x[0,q+k] = yy[j,p].real
#                            q += 1
#                            x[0,q+k] = yy[j,p].imag
#                            q += 1
#                    k += n_states**2
#                Xin.append(list(x.copy()))
#                k = 0
#                for l in range(j+1, yy.shape[0]):
#                    q = 0
#                    for p in labels:
#                        if p%divider == 0:
#                            y[0,q+k] = yy[l,p].real
#                            q += 1
#                        else:
#                            y[0,q+k] = yy[l,p].real
#                            q += 1
#                            y[0,q+k] = yy[l,p].imag
#                            q += 1
#                    k += n_states**2
#                Yin.append(list(y.copy() - prior))
#    def save_data_to_hdf5(filename, data):
#        with h5py.File(filename, 'w') as f:
#            for i, array in enumerate(data):
#                f.create_dataset(f'data_{i}', data=array)
#    save_data_to_hdf5('Xrcpm.h5', Xin) # the input is saved as Xin
#    save_data_to_hdf5('Yrcpm.h5', Yin) # the target values are saved as Yin
#
#
#
#def RCDYN_single(Xin: str,
#        Yin: str,
#        systemType: str,
#        n_states: int,
#        dataCol: int, 
#        xlength: int,
#        time: float,
#        time_step: float,
#        dataPath: str,
#        prior: float):
#    all_files = {}
#    j = 0
#    print('=================================================================')
#    print('prep_input.RCDYN: Grabbing data from "', dataPath, '" directory') 
#    for files in glob.glob(dataPath+'/*'):
#        file_name = os.path.basename(files)
#        all_files[file_name] = np.load(files)
#        j += 1
#
#    if systemType == 'SB':
#        lambNorm = 1.0
#        gammaNorm = 10.0
#        tempNorm = 1.0
#    if systemType == 'FMO':
#        lambNorm = 530.0
#        gammaNorm = 500.0
#        tempNorm = 510.0
#
#    file_count = j
#    
#    print('prep_input.OSTL: Normalizing gamma, lambda and temperature using the following',
#            'normalizing factors in their respective order:', gammaNorm, lambNorm, tempNorm)
##
#    xx = all_files[file_name]
#    tot_length = xx[0:int(round((time+time_step)/time_step) + xlength),:].shape[0]
#    print('prep_input.RCDYN: Number of trajectories =', file_count)
#    m = (tot_length - xlength) * file_count
#
#    labels = []
#    a = 0; b = n_states
#    for i in range(0, n_states):
#        for j in range(a, b):
#            labels.append(j)
#        a += n_states + 1 
#        b += n_states
#    divider = n_states + 1
#    if dataCol == None:
#        x = np.zeros((int(m)*n_states**2, xlength), dtype=float)
#        y = np.zeros((int(m) *n_states**2, 2), dtype=float)
#    else:
#        if (dataCol-1)%divider == 0:                # for diagonal terms, only real terms are considered
#            x = np.zeros((int(m), xlength), dtype=float)
#            y = np.zeros((int(m), 1), dtype=float)
#        else:
#            x = np.zeros((int(m), xlength * 2), dtype=float)  # off-diagonal terms, real and imag 
#            y = np.zeros((int(m), 2), dtype=float)
#    m = 0
#    Xin, Yin = [], []
#    for files in glob.glob(dataPath+'/*'):
#        file_name = os.path.basename(files)
#        df = all_files[file_name]
#        if systemType == 'SB': 
#            x = re.split(r'-', file_name)
#            y = re.split(r'_', x[3])
#            lamb = float(y[0])/lambNorm
#            y = re.split(r'_', x[4])
#            gamma = float(y[0])/gammaNorm
#            y = re.split(r'.n', x[5])
#            temp = float(y[0])/tempNorm
#        else:
#            x = re.split(r'_', file_name)
#            y = re.split(r'-', x[2]) # extracting value of gamma
#            gamma = float(y[1])/gammaNorm
#            y = re.split(r'-', x[3]) # extract value of lambda 
#            lamb = float(y[1])/lambNorm
#            y = re.split(r'-', x[4])
#            x = re.split(r'.npy', y[1]) # extract value of temperature
#            temp = float(x[0])/tempNorm
#        if dataCol != None:
#            for i in range(0, (tot_length - xlength)):
#                yy = df[:, dataCol]
#                if (dataCol -1)%divider == 0:
#                    x[m,:] = yy[i:xlength+i].real
#                    y[m,0] = yy[i+xlength].real
#                else:
#                    x[m,0:xlength] = yy[i:xlength+i].real
#                    y[m,0] = yy[i+xlength].real
#                    x[m,xlength:] = yy[i:xlength+i].imag
#                    y[m,1] = yy[i+xlength].imag
#                m += 1
#        if dataCol == None:
#            for i in range(0, (tot_length - xlength)):
#                yy = df[0:xlength+i+1, 1:n_states**2+1] # excluding the 1st column of time
#                xr = np.zeros((1, xlength + i + 3), dtype=float)
#                xi = np.zeros((1, (xlength + i)*2 + 3), dtype=float)
#                y = np.zeros((1, 2), dtype=float)
#                xr[0, 0] = gamma
#                xr[0, 1] = lamb
#                xr[0, 2] = temp
#                xi[0, 0] = gamma
#                xi[0, 1] = lamb
#                xi[0, 2] = temp
#                for p in labels:
#                    if p%divider == 0:
#                        xr[0,3:] = 0
#                        xr[0,3:] = yy[0:xlength+i,p].real
#                        Xin.append(list(xr.copy()))
#                    else:
#                        xi[0,3:] = 0
#                        xi[0, 3:xlength+i+3] = yy[0:xlength+i,p].real
#                        xi[0, 3+xlength+i:] = yy[0:xlength+i,p].imag
#                        Xin.append(list(xi.copy()))
#                for p in labels:
#                    y[0,:] = 0
#                    if p%divider == 0:
#                        y[0, 0] = yy[xlength+i,p].real - prior
#                        Yin.append(list(y.copy()))
#                    else:
#                        y[0, 0] = yy[xlength+i,p].real - prior
#                        y[0, 1] = yy[xlength+i,p].imag - prior
#                        Yin.append(list(y.copy()))
#    def save_data_to_hdf5(filename, data):
#        with h5py.File(filename, 'w') as f:
#            for i, array in enumerate(data):
#                f.create_dataset(f'data_{i}', data=array)
#    save_data_to_hdf5('Xin.h5', Xin) # the input is saved as Xin
#    save_data_to_hdf5('Yin.h5', Yin) # the target values are saved as Yin

