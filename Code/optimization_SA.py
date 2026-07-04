
############################### Import libraries ###############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import os
import datetime
start_time = datetime.datetime.now()


############################### Create Initial Folder ###############################
# optimization round
round_num  = 1
round_name = 'Round'+str(round_num)
round_num2 = round_num-1
round_name2 = 'Round'+str(round_num2)

# folder for saving optimized results
model_folder = "Results1"
# Check if the directory exists
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(model_folder+'/'+round_name):
    os.makedirs(model_folder+'/'+round_name)
if not os.path.exists(model_folder+'/'+round_name2):
    os.makedirs(model_folder+'/'+round_name2)


############################### Data Preprocess ###############################

data = pd.read_csv(model_folder+'/'+"all_initial_pred.csv")
data = data.fillna(0)
data = np.array(data)

all_input = data[:,0:5] # input of the surrogate model
all_data0 = data[:,5] # output of the surrogate model


plt.figure()
if round_num == 1:
    plt.hist(all_data0,label='initial data')
else:
    plt.hist(all_data0,label='data before')
    plt.hist(all_data0[-20:],label=f'round{round_num2}')
plt.legend()
plt.savefig(model_folder+'/'+round_name2+'/performance-comparison2.png')


############################### Set Parameters ###############################

n_dim = len(all_input[0]) # dimension of the problem
n_model = 5 # number of surrogate models for cross-validation
n_model2 = 20 # number of repeat


#####load all models and store in a dict "models"
path = os.getcwd()
name2=path+'/'+model_folder+'/'+round_name2+'/'
models=dict()
model_list0=[]
for i in range(n_model * n_model2):
    modelname = f'model_SDI{i}'
    model_list0.append(modelname)
    models[modelname]= keras.models.load_model(name2+modelname+'.h5')

# emsemble all models to predict
def ensemble_pred(SDI, n_model):
    pred_all=np.zeros((len(SDI),0))
    SDI[:,0]*=10 ### C element * 10
    for n in range(n_model2):
        for m in range(n_model):
            i = n * n_model + m
            temp=models[model_list0[i]].predict(SDI.reshape(len(SDI),n_dim,1), verbose=0)
            pred_all=np.concatenate((pred_all,temp.reshape(-1,1)),axis=1)
            
    mean = np.mean(pred_all,axis=1).reshape(-1,1)
    std = np.std(pred_all,axis=1).reshape(-1,1)
    
    return np.concatenate((mean, std), axis=1)

def objective_function(x):
    x_proposed=np.array(x).reshape(-1,n_dim,1)
    mean = ensemble_pred(x_proposed, n_model)[:,0]
    all_score = mean.reshape(len(x_proposed))
    std = ensemble_pred(x_proposed, n_model)[:,1]
    uncertainty = std.reshape(len(x_proposed))
    return all_score, uncertainty


############################### Rules for searching new alloys ###############################

import random

# Neighbor function: small random change
def generate_neighbor(current_solution):
    """Generate neighboring solution with random perturbation"""
    perturbation = np.random.uniform(
    low=-np.array(step_sizes),
    high=np.array(step_sizes),
    size=len(current_solution)
    )
    new_solution = current_solution + perturbation
    
    # Ensure values wihtin the boundary
    for i in range(len(new_solution)):
        low, high = bounds[i]
        new_solution[i] = np.clip(new_solution[i], low, high)
    
    # Adust to 100%
    new_solution[4] = 100 - sum(new_solution[:4])
    
    return new_solution

step_sizes = [0.15, 4, 1.5, 1, 0]

bounds = [
    (0, 3),
    (0, 40),
    (0, 15),
    (0, 10),
    (0, 100)
]

############################### Simulated Annealing ###############################
# Simulated Annealing function
def simulated_annealing(current_sol, current_SDI):
    # Algorithm parameters
    initial_temp = 100.0
    cooling_rate = 0.8
    final_temp = 1e-7
    max_iter_per_temp = 10
    
    # Initialize
    best_sol = current_sol.copy()
    best_SDI = current_SDI
    history= []
    
    temp = initial_temp
    iteration = 0
    
    while temp > final_temp:
        for _ in range(max_iter_per_temp):
            # Generate new solution
            new_sol = generate_neighbor(current_sol)
            new_SDI, new_SDI_uncertainty = objective_function(new_sol)
            
            # Calculate energy difference
            delta = new_SDI - current_SDI
            
            # Acceptance criteria
            if delta > 0:  # Accept better solution
                current_sol = new_sol
                current_SDI = new_SDI
                current_SDI_uncertainty = new_SDI_uncertainty
                if new_SDI > best_SDI:
                    best_sol = new_sol.copy()
                    best_SDI = new_SDI
            else:  # Probabilistically accept worse solution
                if np.random.rand() < np.exp(delta / temp):
                    current_sol = new_sol
                    current_SDI = new_SDI
                    current_SDI_uncertainty = new_SDI_uncertainty
            
            hist = np.hstack([temp, current_sol, current_SDI, best_sol, best_SDI, current_SDI_uncertainty])
            history.append(hist)
            iteration += 1
            print(f"Iteration {iteration}, Temp {temp}, Current SDI {current_SDI}, Best SDI {best_SDI}")
            
        # Cooling
        temp *= cooling_rate
        
    np.save(os.path.join(model_folder, round_name, 'history15.npy'), history)
    return best_sol, best_SDI


elements = ['C','Mn','Al','Si','Fe']
initial_pred, _ = objective_function(all_input)

# Run optimization
optimal_composition, max_SDI = simulated_annealing(all_input[np.argmax(initial_pred[0])],np.max(initial_pred[0]))

end_time = datetime.datetime.now()

# Display results
print("\nOptimized Alloy Composition:")
for element, percentage in zip(elements, optimal_composition):
    print(f"{element}: {percentage:.2f}%")
print(f"Predicted SDI: {max_SDI}")
print("\nStart time:", start_time, "\n  End time:", end_time)

