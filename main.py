# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license

import os
import time
import argparse
from datetime import datetime
from openai import OpenAI

from graph_understanding.graph_understanding import GraphDatasetUnderstanding
from graph_comparison.graph_comparison import GraphDatasetComparison
from knowledge_retrieval.knowledge_retrieval import KnowledgeRetrieval
from knowledge_retrieval.knowledge_estimation import KnowledgeEstimation
from model_refinement.model_refinement import ModelRefinement
from model_refinement.config import design_choice_translate, design_dimensions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Actor', 'AmazonComputers', 'AmazonPhoto', 'CiteSeer', 'CoauthorCS', 
                                                                        'Cora', 'Cornell', 'DBLP', 'PubMed', 'Texas', 'Wisconsin', 'TU_COX2', 'TU_DD', 
                                                                        'TU_IMDB-BINARY', 'TU_IMDB-MULTI', 'TU_NCI1', 'TU_NCI109', 'TU_PROTEINS', 
                                                                        'TU_PTC_FM', 'TU_PTC_FR', 'TU_PTC_MM', 'TU_PTC_MR'],
                        help='Dataset name. Default Cora.')
    parser.add_argument('--task', type=str, default='node_classification', choices=['node_classification', 'link_prediction', 'graph_classification'],
                        help='Task: node_classification, link_prediction, graph_classification. Default node_classification.')
    
    # Search strategy
    parser.add_argument('--search_strategy', type=str, default='kg_controller', choices=['kg_controller'],
                        help='Search strategy. Default kg_controller.')
    parser.add_argument('--ensembling', type=str, default='bayesian_update', choices=['bayesian_update'],
                        help='Knowledge weaving method. Default bayesian_update.')
    parser.add_argument('--initial_strategy', type=str, default='weighted_average', choices=['weighted_average', 'majority_vote', 'best'],
                        help='Initial proposal strategy: weighted_average, majority_vote, best. Fixed to best if ensembling is separated. Default weighted_average.')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of iterations (trials) for the model design refinement. Default 30.')

    # Similarity Metric
    parser.add_argument('--similarity_threshold', type=float, default=None,
                        help='Similarity threshold to consider as prior knowledge. Default None.')
    parser.add_argument('--min_top_s', type=int, default=1,
                        help='Minimum number of similar benchmark datasets to consider as prior knowledge. Default 1.')
    parser.add_argument('--similarity_metric', type=str, default='kendall', choices=['kendall', 'LLM'],
                        help='Initial similarity metric: kendall, LLM. Default kendall.')
    parser.add_argument('--dynamic_similarity', type=str, default='bayesian_update',
                        help='Dynamic similarity metric. Default bayesian_update.')
    parser.add_argument('--use_estimator', action='store_true', default=False,
                        help='Allow GNN-based modification gain predictor to replace knowledge retrieval. Default False.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='Learning rate for dynamic similarity. Default 0.1.')
    parser.add_argument('--window', type=int, default=None,
                        help='Window size for dynamic similarity. Default None.')

    # LLM Prompting
    parser.add_argument('--top_ratio', type=str, default='best',
                        help='Top ratio of the best-performed models to consider on each benchmark. 0.05 -> 5%. Default best.')
    
    args = parser.parse_args()

    # Assuming descriptions and other necessary data are already defined
    unseen_dataset_name = args.dataset
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    response_save_path = os.path.join(os.getcwd(), 'responses', f'{unseen_dataset_name}',
                                      f'{unseen_dataset_name}_{args.search_strategy}_{current_time}')
    os.makedirs(response_save_path, exist_ok=True)

    if args.task in ['node_classification', 'link_prediction']:
        benchmark_datasets = ['Actor', 'AmazonComputers', 'AmazonPhoto', 'CiteSeer', 'CoauthorCS', 'Cora', 'Cornell', 'DBLP', 'PubMed', 'Texas', 'Wisconsin']
    elif args.task == 'graph_classification':
        benchmark_datasets = ['TU_COX2', 'TU_DD', 'TU_IMDB-BINARY', 'TU_IMDB-MULTI', 'TU_NCI1', 'TU_NCI109', 'TU_PROTEINS', 'TU_PTC_FM', 'TU_PTC_FR', 'TU_PTC_MM', 'TU_PTC_MR']
    else:
        raise ValueError('Unknown task: {}'.format(args.task))
    benchmark_datasets = [dataset for dataset in benchmark_datasets if args.dataset not in dataset]

    dataset_dir = 'GraphGym/run/datasets/'

    # --------------------------------------------------------------------------------
    # Step 1: Graph Dataset Understanding
    # Process unseen dataset
    understanding_module = GraphDatasetUnderstanding(unseen_dataset_name,
                                                     task=args.task,
                                                     root_dir=dataset_dir,
                                                     no_statistics=False,
                                                     use_semantic=True,
                                                     is_bench=False)
    unseen_dataset_description = understanding_module.process()

    # Process benchmark datasets
    benchmark_dataset_descriptions = {}
    for benchmark_dataset_name in benchmark_datasets:
        benchmark_dataset_understanding = GraphDatasetUnderstanding(benchmark_dataset_name,
                                                                    task=args.task,
                                                                    root_dir=dataset_dir,
                                                                    no_statistics=False,
                                                                    use_semantic=True,
                                                                    is_bench=True)
        benchmark_dataset_descriptions[benchmark_dataset_name] = benchmark_dataset_understanding.process()

    # --------------------------------------------------------------------------------
    # Step 2: Graph Dataset Comparison
    # Initialize OpenAI client
    with open('key.txt', 'r') as file:
        client = OpenAI(api_key=file.read().strip())

    # Initialize the KnowledgeRetrieval module
    knowledge_retrieval = KnowledgeRetrieval(task=args.task)
    knowledge_estimatior = KnowledgeEstimation(task=args.task, buffer_size=args.window)
    
    # Compare the unseen dataset with benchmark datasets
    dataset_comparison = GraphDatasetComparison(design_choice_translate=design_choice_translate,
                                                design_dimensions=design_dimensions[args.task],
                                                knowledge_retrieval=knowledge_retrieval,
                                                top_ratio=args.top_ratio,
                                                response_save_path=response_save_path)
    if args.similarity_metric == 'kendall':
        benchmark_similarity = dataset_comparison.calculate_kendall_rank_similarities(dataset_dir,
                                                                                      benchmark_datasets + [unseen_dataset_name],
                                                                                      unseen_dataset_name)
    elif args.similarity_metric == 'LLM':
        similarity_response = dataset_comparison.compare_datasets(client,
                                                                unseen_dataset_description,
                                                                benchmark_dataset_descriptions)
        benchmark_similarity = dataset_comparison.to_dict(similarity_response, benchmark_datasets)
    else:
        raise ValueError('Unknown similarity metric: {}'.format(args.similarity_metric))

    # Get the names of the top-s benchmarks
    top_benchmarks, args.similarity_threshold, args.min_top_s = dataset_comparison.determine_similar_datasets(benchmark_similarity,
                                                                                                              args.similarity_threshold,
                                                                                                              args.min_top_s)
    print(f"Top {args.min_top_s} similar benchmarks (>= {args.similarity_threshold}): {top_benchmarks}")
    with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
        file.write(f"Top {args.min_top_s} similar benchmarks (>= {args.similarity_threshold}): {top_benchmarks}\n")

    # --------------------------------------------------------------------------------
    model_scientist = ModelRefinement(args.task, 
                                      args.min_top_s, 
                                      args.eta,
                                      args.max_iter, 
                                      knowledge_retrieval, 
                                      knowledge_estimatior,
                                      args.window)
    selected_initial_proposal, selected_final_proposal, selected_accuracy_history = model_scientist.recommend_initial_proposal_and_refine(args.search_strategy,
                                                                                                                                          args.ensembling,
                                                                                                                                          args.initial_strategy,
                                                                                                                                          args.dynamic_similarity,
                                                                                                                                          args.use_estimator,
                                                                                                                                          args.dataset,
                                                                                                                                          top_benchmarks,
                                                                                                                                          response_save_path)
    print(f"\n\nSelected Initial Proposal: {selected_initial_proposal}")
    print(f"Selected Final Proposal: {selected_final_proposal}")
    with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
        file.write(f"\n\nSelected Initial Proposal: {selected_initial_proposal}\n")
        file.write(f"Selected Final Proposal: {selected_final_proposal}\n")
    
    # Define the checkpoints
    checkpoints = [10, 30, 50, 70, 100]

    # Iterate over the checkpoints and print/save the best-so-far accuracy
    for checkpoint in checkpoints:
        if args.max_iter >= checkpoint and len(selected_accuracy_history) >= checkpoint:
            # Get the accuracies up to the current checkpoint
            accuracies_up_to_checkpoint = selected_accuracy_history[:checkpoint]
            # Find the best-so-far accuracy
            best_so_far = max(accuracies_up_to_checkpoint, key=lambda x: x[0])
            print(f"Best-so-far accuracy at iteration {checkpoint}: {best_so_far[0]} (std: {best_so_far[1]})")
            with open(os.path.join(response_save_path, f"refinement_summary.txt"), 'a') as file:
                file.write(f"Best-so-far accuracy at iteration {checkpoint}: {best_so_far[0]} (std: {best_so_far[1]})\n")


if __name__ == "__main__":
    main()

