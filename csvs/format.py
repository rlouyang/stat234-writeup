#!/usr/bin/env python

import pandas as pd

def output(file, keys):
    df = pd.read_csv(file, na_values=['sort of. this is a DDQN', 'N/A'])
    df.columns = list(map(str.lower, df.columns))
    df.columns = list(map(lambda x: x.replace(' ', '_'), df.columns))
    for k in keys:
        if k in df.columns:
            del df[k]        
    if 'has_target_network' in df.columns:
        df.columns = [x if x != 'has_target_network' else 'target' for x in df.columns]
    if 'initial_learning_rate' in df.columns:
        df.columns = [x if x != 'initial_learning_rate' else 'lr' for x in df.columns]
    if 'learning_rate_annealing' in df.columns:
        df.columns = [x if x != 'learning_rate_annealing' else 'lr_anneal' for x in df.columns]
    print(df.columns)
    result = df.sort_values(by=['game', 'mean'], ascending=False).groupby('game').head(15)
    with open(file.split('.')[0] + '.tex', 'w') as f:
        f.write(result.to_latex(index=False, na_rep='N/A'))

# output('model_selection_100k.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'initial_learning_rate', 'weight_decay'])

# output('model_selection_ddqn_pca_4.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'initial_learning_rate', 'weight_decay'])

output('model_selection_DDQN_PCA_5.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'initial_learning_rate', 'weight_decay', 'num_parameters'])

output('model_selection_DDQN_PCA_6.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'initial_learning_rate', 'weight_decay', 'num_parameters'])

output('grid_search_v1_10k.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'processing', 'optimizer', 'batch_size', 'gamma', 'frame_skip', 'update_frequency', 'max', 'min', 'num_parameters'])

output('grid_search_v2_50k.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'annealing', 'optimizer', 'loss_function', 'layer_sizes', 'num_parameters'])

output('grid_search_v3_100k.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'annealing', 'optimizer', 'loss_function', 'layer_sizes', 'weight_decay', 'model', 'num_parameters'])

# output('final_v1_100k.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'layer_sizes', 'initial_learning_rate', 'weight_decay'])

output('final_v2_100k.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'layer_sizes', 'initial_learning_rate', 'weight_decay'])

# output('dqn_pca_3.csv', ['key', '25', '50', '75', 'agent', 'n_trains', 'target_update', 'frame_skip', 'processing', 'update_frequency', 'gamma', 'batch_size', 'annealing', 'optimizer', 'loss_function', 'initial_learning_rate', 'weight_decay'])
