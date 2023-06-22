from datetime import datetime
import os
import pandas as pd
import random 
import glob

LABELS = {'AD': 1, 'NC': 0}

def generate_csv(path, train, valid, test, output):
    random.seed(42)
    if not train + valid + test == 100:
        raise AssertionError('Total values of the split must equal to 100!')

    dataset_name = os.path.basename(path)
    
    ad_folder = glob.glob(os.path.join(path, 'AD/*'))
    nc_folder = glob.glob(os.path.join(path, 'NC/*'))

    splits = get_splits(ad_folder, nc_folder, train, valid)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    dfs, paths = [], []
    for split, items in splits.items():
        data = {
            'split': [split] * len(items),
            'label': [LABELS.get(item.split('/')[-2], 0) for item in items],
            'num_scans': [1] * len(items),
            'scan1': items
        }
            
        df = pd.DataFrame(data)
        dfs.append(df)
        
        filename = f'{dataset_name}_{split}_{current_time}_dataset.csv'
        filepath = os.path.join(output, filename)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath}")
        paths.append(filepath)
        
    all_data_df = pd.concat(dfs)
    all_filename = f'{dataset_name}_all_{current_time}_dataset.csv'
    all_filepath = os.path.join(output, all_filename)
    all_data_df.to_csv(all_filepath, index=False)
    print(f"Saved {all_filepath}")
    
    return paths[0], paths[1], paths[2]

def get_splits(ad_folder, nc_folder, train, valid):
    ad_splits = split_data(ad_folder, train, valid)
    nc_splits = split_data(nc_folder, train, valid)

    splits = {
        'train': ad_splits['train'] + nc_splits['train'],
        'val': ad_splits['val'] + nc_splits['val'],
        'test': ad_splits['test'] + nc_splits['test']
    }

    return splits

def split_data(data, train, valid):
    index_train = int(len(data) * float(train/100))
    index_valid = int(len(data) * float((train + valid)/100))

    splits = {
        'train': data[:index_train],
        'val': data[index_train:index_valid],
        'test': data[index_valid:]
    }

    return splits
