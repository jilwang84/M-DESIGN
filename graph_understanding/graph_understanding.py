# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file will serve as the main module to orchestrate the understanding of graph datasets.

from .summarizer import GraphSummarizer
from .loader import load_pyg
import json


class GraphDatasetUnderstanding:
    def __init__(self, dataset_name, task, user_description=None, root_dir='GraphGym/run/datasets/', 
                 no_statistics=False, use_semantic=False, is_bench=False, seed=42):
        """
        Initializes the GraphDatasetUnderstanding class.
        :param dataset_name: Name of the dataset to understand.
        :param user_description: Description of the dataset provided by the user.
        :param root_dir: Root directory where the dataset is stored.
        :param no_statistics: Flag to indicate whether to compute statistics for the dataset.
        :param use_semantic: Flag to indicate whether to use semantic understanding for the dataset.
        :param seed: Seed value for reproducibility.
        """
        self.root_dir = root_dir
        self.no_statistics = no_statistics
        self.is_bench = is_bench
        self.dataset_name = dataset_name
        self.seed = seed

        if task == 'node_classification':
            task = 'NC'
        elif task == 'link_prediction':
            task = 'LP'
        elif task == 'graph_classification':
            task = 'GC'
        else:
            raise ValueError(f"Unknown task: {task}")
        self.if_GC = task == 'GC'

        # Load predefined descriptions
        if is_bench:
            predefined_descriptions_path='graph_understanding/predefined_descriptions (bench).json'
        else:
            predefined_descriptions_path='graph_understanding/predefined_descriptions (unseen).json'
        with open(predefined_descriptions_path, 'r') as file:
            predefined_semantic_descriptions = json.load(file)

        if is_bench:
            if use_semantic:
                self.semantic_description = user_description or predefined_semantic_descriptions.get(f"{task}-{self.dataset_name}", None)
                if self.semantic_description is None:
                    raise ValueError(f"Semantic description for benchmark dataset {self.dataset_name} is not provided.")
            else:
                self.semantic_description = None
        else:
            default_msg = "However, the user does not provide a semantic description for the target graph dataset (unseen), " \
                          "please understand it based entirely on the following graph properties. "
            self.semantic_description = user_description or predefined_semantic_descriptions.get(f"{task}-{self.dataset_name}", default_msg)

    def process(self):
        """
        Processes the graph dataset to understand it through various descriptions.
        :return: Combined description of the dataset suitable for LLM prompting.
        """
        if self.no_statistics:
            description = ""
        else:
            dataset = load_pyg(self.dataset_name, self.root_dir)
            summarizer = GraphSummarizer(dataset, self.dataset_name, self.seed)
            summarizer.summarize(self.root_dir)
            description = summarizer.to_natural_language(self.is_bench, self.if_GC)
        
        if self.semantic_description:
            description += self.semantic_description
        
        if description == "":
            raise ValueError("No description is generated for the dataset.")

        return description

