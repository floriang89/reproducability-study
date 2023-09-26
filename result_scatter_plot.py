# Scatter plot of results
import matplotlib.pyplot as plt
import pandas as pd

labels  = ['Tax+CS', 'Tax+T-LSTM', 'Tax+CS+T-LSTM']
results_acc = {
    'Helpdesk Accuracy (higher is better)': {
        'Reported results time-matters': [0.713, 0.718, 0.724],
        'Reproduced results Grid Search': [0.713, 0.699, 0.699],
        'Reproduced results Grid Search, Fixed One-Hot encoding': [0.738, 0.722, 0.722],
        'Reproduced results Optuna': [0.713, 0.539, 0.539],
        'Reproduced results Optuna, Fixed One-Hot encoding': [0.713, 0.714, 0.714]
    },
    'BPI 12 W Accuracy (higher is better)': {
        'Reported results time-matters': [0.757, 0.693, 0.778],
        'Reproduced results Grid Search': [0.837, 0.356, 0.356],
        'Reproduced results Grid Search, Fixed One-Hot encoding': [0.702, 0.783, 0.783],
        'Reproduced results Optuna': [0.773, 0.652, 0.652],
        'Reproduced results Optuna, Fixed One-Hot encoding': [0.759, 0.639, 0.639]
    },
    'Helpdesk MAE (lower is better)': {
        'Reported results time-matters': [2.87, 3.01, 2.94],
        'Reproduced results Grid Search': [2.92, 3.17, 3.17],
        'Reproduced results Grid Search, Fixed One-Hot encoding': [2.83, 2.96, 2.96],
        'Reproduced results Optuna': [2.98, 3.01, 3.01],
        'Reproduced results Optuna, Fixed One-Hot encoding': [2.89, 2.94, 2.94]
    },
    'BPI 12 W MAE (lower is better)': {
        'Reported results time-matters': [0.88, 0.88, 0.90],
        'Reproduced results Grid Search': [0.86, 0.91, 0.91],
        'Reproduced results Grid Search, Fixed One-Hot encoding': [0.86, 0.87, 0.87],
        'Reproduced results Optuna': [0.88, 0.89, 0.89],
        'Reproduced results Optuna, Fixed One-Hot encoding': [0.86, 0.91, 0.91]
    }
}

fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharey=False)
#axs[0].grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
#axs[1].grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)

markers = ("o", "v", "s", "X", "P")
colors = ("#4e79a7", "#f28e2b", "#e15759", "black", "#59a14f")

i = 0
j = 0
legend_flag = True

for titel, results in results_acc.items():
    
    axs[i][j].grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7)
    axs[i][j].set_title(titel, loc='left',  fontsize=18)
    axs[i][j].tick_params(axis='x', labelsize=18)
    axs[i][j].tick_params(axis='y', labelsize=18)
    
    k = 0
    for result, value in results.items():
        if legend_flag:
            axs[i][j].scatter(labels, value, s=400, alpha=0.5, marker=markers[k], color=colors[k], label= result)
        else:
            axs[i][j].scatter(labels, value, s=400, alpha=0.5, marker=markers[k], color=colors[k])
        k+=1 
    if j == len(axs)-1:
        j=0
        i+=1
    else:
        j+=1
    legend_flag = False

fig.legend(loc='outside lower center', ncols=3, fontsize=12)
fig.tight_layout(pad=5.0)
fig.suptitle('Results', fontsize=26)

plt.savefig('results_scatter.png', dpi='figure', format='png',pad_inches=0.5,bbox_inches='tight')