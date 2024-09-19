# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:44:53 2024

@author: hazem
"""

import os

def create_directories(base_dir=""):
    datasets_dir = os.path.join(base_dir, 'datasets')
    for dataset in ['Arxiv', 'CNNDM', 'Pubmed']:
        dataset_dir = os.path.join(datasets_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        for split in ['train', 'test', 'validation']:
            split_dir = os.path.join(dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            for folder in ['Sen_sim', 'Sent_adj', 'WS_edges', 'WW_edges']:
                folder_dir = os.path.join(split_dir, folder)
                os.makedirs(folder_dir, exist_ok=True)
                
    datasets_dir = os.path.join(base_dir, 'Graphs')
    for dataset in ['Arxiv', 'CNNDM', 'Pubmed']:
        dataset_dir = os.path.join(datasets_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        for split in ['train', 'test', 'validation']:
            split_dir = os.path.join(dataset_dir, split)
            os.makedirs(split_dir, exist_ok=True)

def main():
    create_directories()

main()