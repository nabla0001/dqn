"""
Contains all settings for DQN experiments.

-- Agent settings (learning parameters)
-- Environment settings (cropping bounding box, etc.)
"""

preproc = {
    'MsPacman-v3':  {'bbox':  (0,0,170,160),'D': [64,64,4]},
    'Pong-v3':      {'bbox':  (35,0,157,160),'D': [84,84,4]},
    'Boxing-v3':    {'bbox':  (35,30,145,100),'D': [84,84,4]},
}
