# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import argparse
import pandas as pd
import sqlite3


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='node_classification',
                    help='Task: node_classification, link_prediction, graph_classification. Default node_classification.')
args = parser.parse_args()

task_name = {
    'node_classification': 'node',
    'link_prediction': 'link',
    'graph_classification': 'graph'
}

if task_name[args.task] == 'node' or task_name[args.task] == 'link':
    datasets = ['Actor', 'AmazonComputers', 'AmazonPhoto', 'CiteSeer', 'CoauthorCS', 'Cora', 'Cornell', 'DBLP', 'PubMed', 'Texas', 'Wisconsin']
else:
    datasets = ['TU_COX2', 'TU_DD', 'TU_IMDB-BINARY', 'TU_IMDB-MULTI', 'TU_NCI1', 'TU_NCI109', 'TU_PROTEINS', 'TU_PTC_FM', 'TU_PTC_FR', 'TU_PTC_MM', 'TU_PTC_MR']

for dataset in datasets:
    # Load the uploaded CSV file
    file_path = f"knowledge_base/{task_name[args.task]}/{dataset.lower()}/agg/test_best.csv"
    data = pd.read_csv(file_path)

    # Create an SQLite database in-memory
    db_path = f"knowledge_base/{task_name[args.task]}/{dataset.lower()}/{dataset}.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table for model records
    if args.task == 'node_classification':
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_records (
            neigh TEXT,
            norm TEXT,
            agg TEXT,
            comb TEXT,
            l_mp TEXT,
            stage TEXT,
            accuracy REAL,
            accuracy_std REAL,
            PRIMARY KEY (neigh, norm, agg, comb, l_mp, stage)
        )
        ''')
        # Insert the data into the SQLite table
        insert_query = '''
        INSERT OR IGNORE INTO model_records (neigh, norm, agg, comb, l_mp, stage, accuracy, accuracy_std)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = data[["neigh", "norm", "agg", "comb", "l_mp", "stage", "accuracy", "accuracy_std"]].values.tolist()
    elif args.task == 'link_prediction' or args.task == 'graph_classification':
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_records (
            decode TEXT,
            norm TEXT,
            agg TEXT,
            comb TEXT,
            l_mp TEXT,
            stage TEXT,
            accuracy REAL,
            accuracy_std REAL,
            PRIMARY KEY (decode, norm, agg, comb, l_mp, stage)
        )
        ''')
        # Insert the data into the SQLite table
        insert_query = '''
        INSERT OR IGNORE INTO model_records (decode, norm, agg, comb, l_mp, stage, accuracy, accuracy_std)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = data[["decode", "norm", "agg", "comb", "l_mp", "stage", "accuracy", "accuracy_std"]].values.tolist()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    cursor.executemany(insert_query, records)
    conn.commit()

