import os
import numpy as np
import pickle
import glob
import argparse

from tf_utils import plot_std

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


# ----- parse command line -----
parser = argparse.ArgumentParser()
parser.add_argument('--path','-d', type=str, default='.',
                  help='Checkpoint folder.')
parser.add_argument('--output','-o', type=str, default='plot.pdf',
                  help='Output file name.')
FLAGS, _ = parser.parse_known_args()
# ------------------------------

files = sorted(glob.glob(os.path.join(FLAGS.path,'*.p')),key=os.path.getmtime)
print files

# merge results
combined = {}
for f in files:
    d = pickle.load(open(f,'rb'))

    for key in d.keys():

        if key in combined.keys():
            combined[key] = np.concatenate((combined[key],d[key]),axis=0)
        else:
            combined[key] = d[key]

f,((ax1,ax2)) = plt.subplots(2,1)

plot_std(combined['x'],ax2,combined['n_frames'],axis=1,col='r')
plot_std(combined['x'],ax1,combined['score'],axis=1,col='b')

ax2.set_ylabel('Game length [mean across 20 episodes]')
ax1.set_ylabel('Score [mean across 20 episodes]')
ax2.set_xlabel('# frames')
ax1.set_xlabel('# frames')

# add chance
ax1.axhline(0.1,color='k',linestyle='--')
ax2.axhline(2378,color='k',linestyle='--')

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(FLAGS.path,FLAGS.output))
