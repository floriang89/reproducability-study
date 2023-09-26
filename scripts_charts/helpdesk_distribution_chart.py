# Activity distribution of helpdesk dataset to show skewedness
import matplotlib.pyplot as plt
import pandas as pd

spamread=pd.read_csv('data/processed/helpdesk.csv',delimiter=',',quotechar='|', index_col=False)
elems_per_fold = int(round(len(spamread) / 3)) # Training set is 2/3 of the total dataset
training = spamread[:elems_per_fold*2]
df_groups = training.groupby('ActivityID')['CaseID'].count()
labels = df_groups.index.to_list()
data = df_groups.to_list()

plt.figure(figsize = (10, 5))
plt.xticks(range(len(data)), labels)
plt.xlabel('Activity IDs', fontsize=18)
plt.ylabel('Number of activities', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.7,zorder=0)
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='x', alpha=0, zorder=1)
barplot = plt.bar(range(len(data)), data, color='#4e79a7', zorder=2)
plt.title('Helpdesk Training Set Activity Distribution', fontsize=18)
plt.bar_label(barplot,label_type='edge', padding = 5, fontsize=18)
plt.bar(range(len(data)), data) 
plt.ylim([0,3200])
plt.savefig('helpdesk_training_set_distribution.png', dpi='figure', format='png',pad_inches=0.5,bbox_inches='tight')