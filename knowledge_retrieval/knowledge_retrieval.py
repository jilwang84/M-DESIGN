# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

import copy
from pathlib import Path
import sqlite3
from collections import Counter

from .candidate_evaluator import CandidateEvaluationConfig, CandidateEvaluator


class KnowledgeRetrieval:
    def __init__(
        self,
        task,
        base_db_path=None,
        evaluation_mode="auto",
        graphgym_root="GraphGym",
        candidate_output_dir="outputs/candidate_runs",
        gpu_id=0,
        repeat=3,
        max_epoch=None,
    ):
        self.task = task
        self.evaluation_mode = evaluation_mode
        if evaluation_mode not in {"auto", "database", "train"}:
            raise ValueError("evaluation_mode must be one of: auto, database, train")
        if task == 'node_classification':
            task_dir = "node"
        elif task == 'link_prediction':
            task_dir = "link"
        elif task == 'graph_classification':
            task_dir = "graph"
        else:
            raise ValueError(f"Unknown task: {task}")
        if base_db_path is None:
            self.base_db_path = Path("knowledge_retrieval/knowledge_base") / task_dir
        else:
            base_path = Path(base_db_path)
            self.base_db_path = base_path / task_dir if (base_path / task_dir).exists() else base_path
        evaluator_config = CandidateEvaluationConfig(
            graphgym_root=Path(graphgym_root).resolve(),
            output_root=Path(candidate_output_dir).resolve(),
            gpu_id=gpu_id,
            repeat=repeat,
            max_epoch=max_epoch,
        )
        self.candidate_evaluator = CandidateEvaluator(task, evaluator_config)

    def has_dataset(self, dataset):
        return self._db_path(dataset).exists()

    def _db_path(self, dataset):
        return self.base_db_path / dataset.lower() / f"{dataset}.db"

    def retrieve_model(self, dataset, model):
        if self.task == 'node_classification':
            query = '''
            SELECT accuracy, accuracy_std FROM model_records
            WHERE neigh = ? AND norm = ? AND agg = ? AND comb = ? AND l_mp = ? AND stage = ?
            '''
            conn = sqlite3.connect(self._db_path(dataset))
            cursor = conn.cursor()
            cursor.execute(query, (model["neigh"], model["norm"], model["agg"], model["comb"], model["l_mp"], model["stage"]))
            result = cursor.fetchone()
            conn.close()
        elif self.task == 'link_prediction' or self.task == 'graph_classification':
            query = '''
            SELECT accuracy, accuracy_std FROM model_records
            WHERE decode = ? AND norm = ? AND agg = ? AND comb = ? AND l_mp = ? AND stage = ?
            '''
            conn = sqlite3.connect(self._db_path(dataset))
            cursor = conn.cursor()
            cursor.execute(query, (model["decode"], model["norm"], model["agg"], model["comb"], model["l_mp"], model["stage"]))
            result = cursor.fetchone()
            conn.close()
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return result

    def evaluate_model(self, dataset, model):
        """Evaluate a candidate on the target dataset.

        In ``auto`` mode, M-DESIGN first uses the released database if the target
        dataset is part of the model bank. If no database record is available, it
        falls back to GraphGym training/evaluation, using GPU when available.
        """
        if self.evaluation_mode in {"auto", "database"} and self.has_dataset(dataset):
            result = self.retrieve_model(dataset, model)
            if result is not None:
                return result
            if self.evaluation_mode == "database":
                raise LookupError(f"No database record found for {dataset}: {model}")

        if self.evaluation_mode == "database":
            raise FileNotFoundError(f"No released database found for target dataset {dataset}")

        return self.candidate_evaluator.evaluate(dataset, model)
    
    def retrieve_top_n_best(self, dataset, n=5):
        if self.task == 'node_classification':
            query = '''
            SELECT neigh, norm, agg, comb, l_mp, stage, accuracy, accuracy_std
            FROM model_records
            ORDER BY accuracy DESC
            LIMIT ?
            '''
        elif self.task == 'link_prediction' or self.task == 'graph_classification':
            query = '''
            SELECT decode, norm, agg, comb, l_mp, stage, accuracy, accuracy_std
            FROM model_records
            ORDER BY accuracy DESC
            LIMIT ?
            '''
        else:
            raise ValueError(f"Unknown task: {self.task}")
        conn = sqlite3.connect(self._db_path(dataset))
        cursor = conn.cursor()
        cursor.execute(query, (n,))
        results = cursor.fetchall()
        conn.close()

        return results
    
    def analyze_most_frequent(self, dataset, threshold=0.05):
        conn = sqlite3.connect(self._db_path(dataset))
        cursor = conn.cursor()

        # Step 1: Determine the Top t% accuracy threshold
        cursor.execute("SELECT accuracy FROM model_records ORDER BY accuracy DESC")
        all_accuracies = [row[0] for row in cursor.fetchall()]
        top_t_percent_threshold = all_accuracies[int(len(all_accuracies) * threshold)]

        # Step 2: Fetch the Top t% best-performed models
        if self.task == 'node_classification':
            query_top_t_percent = '''
            SELECT neigh, norm, agg, comb, l_mp, stage
            FROM model_records
            WHERE accuracy >= ?
            '''
        elif self.task == 'link_prediction' or self.task == 'graph_classification':
            query_top_t_percent = '''
            SELECT decode, norm, agg, comb, l_mp, stage
            FROM model_records
            WHERE accuracy >= ?
            '''
        else:
            raise ValueError(f"Unknown task: {self.task}")
        cursor.execute(query_top_t_percent, (top_t_percent_threshold,))
        top_t_percent_models = cursor.fetchall()

        # Step 3: Analyze the most frequent choice for each design dimension
        if self.task == 'node_classification':
            neigh_counter = Counter([model[0] for model in top_t_percent_models])
            norm_counter = Counter([model[1] for model in top_t_percent_models])
            agg_counter = Counter([model[2] for model in top_t_percent_models])
            comb_counter = Counter([model[3] for model in top_t_percent_models])
            l_mp_counter = Counter([model[4] for model in top_t_percent_models])
            stage_counter = Counter([model[5] for model in top_t_percent_models])

            # Identify the most frequent choices
            most_frequent_choices = {
                "neigh": neigh_counter.most_common(1)[0],
                "norm": norm_counter.most_common(1)[0],
                "agg": agg_counter.most_common(1)[0],
                "comb": comb_counter.most_common(1)[0],
                "l_mp": l_mp_counter.most_common(1)[0],
                "stage": stage_counter.most_common(1)[0],
            }
        elif self.task == 'link_prediction' or self.task == 'graph_classification':
            decode_counter = Counter([model[0] for model in top_t_percent_models])
            norm_counter = Counter([model[1] for model in top_t_percent_models])
            agg_counter = Counter([model[2] for model in top_t_percent_models])
            comb_counter = Counter([model[3] for model in top_t_percent_models])
            l_mp_counter = Counter([model[4] for model in top_t_percent_models])
            stage_counter = Counter([model[5] for model in top_t_percent_models])

            # Identify the most frequent choices
            most_frequent_choices = {
                "decode": decode_counter.most_common(1)[0],
                "norm": norm_counter.most_common(1)[0],
                "agg": agg_counter.most_common(1)[0],
                "comb": comb_counter.most_common(1)[0],
                "l_mp": l_mp_counter.most_common(1)[0],
                "stage": stage_counter.most_common(1)[0],
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")

        conn.close()

        return most_frequent_choices
    
    def get_all_one_step_modifications(self, dataset, current_model):
        # Connect to the specific dataset's database
        conn = sqlite3.connect(self._db_path(dataset))
        cursor = conn.cursor()

        modifications = []

        if self.task == 'node_classification':
            fields = ['neigh', 'norm', 'agg', 'comb', 'l_mp', 'stage']
        elif self.task == 'link_prediction' or self.task == 'graph_classification':
            fields = ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage']
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Iterate over each field that can be modified
        for field in fields:
            current_value = current_model[field]

            # Retrieve all unique values for the field from the database, excluding the current value
            query = f'''
            SELECT DISTINCT {field}
            FROM model_records
            WHERE {field} != ?
            '''
            cursor.execute(query, (current_value,))
            possible_values = [row[0] for row in cursor.fetchall()]

            # Generate a modified model for each possible value and retrieve its performance
            for value in possible_values:
                modified_model = copy.deepcopy(current_model)
                modified_model[field] = value

                # Retrieve the accuracy and accuracy_std for the modified model
                if self.task == 'node_classification':
                    performance_query = '''
                    SELECT accuracy, accuracy_std
                    FROM model_records
                    WHERE neigh = ? AND norm = ? AND agg = ? AND comb = ? AND l_mp = ? AND stage = ?
                    '''
                    cursor.execute(performance_query, (
                        modified_model['neigh'], modified_model['norm'], modified_model['agg'],
                        modified_model['comb'], modified_model['l_mp'], modified_model['stage']
                    ))
                elif self.task == 'link_prediction' or self.task == 'graph_classification':
                    performance_query = '''
                    SELECT accuracy, accuracy_std
                    FROM model_records
                    WHERE decode = ? AND norm = ? AND agg = ? AND comb = ? AND l_mp = ? AND stage = ?
                    '''
                    cursor.execute(performance_query, (
                        modified_model['decode'], modified_model['norm'], modified_model['agg'],
                        modified_model['comb'], modified_model['l_mp'], modified_model['stage']
                    ))
                else:
                    raise ValueError(f"Unknown task: {self.task}")
                performance = cursor.fetchone()

                # Only include valid configurations with performance data
                if performance:
                    accuracy, accuracy_std = performance
                    modifications.append((modified_model, accuracy, accuracy_std))

        # Close the connection
        conn.close()

        return modifications

