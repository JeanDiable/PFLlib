#!/usr/bin/env python3
"""
Generate personalized task sequences for Personalized Federated Task-Incremental Learning (PFTIL).

This script generates task sequences following a rotating order pattern:
- Client 0: (0,1), (2,3), (4,5), ...
- Client 1: (2,3), (4,5), (6,7), ...
- Client 2: (4,5), (6,7), (8,9), ...

Usage:
    python generate_task_sequence.py -nc 5 -ntpc 10 -cpt 2 -tc 10 -o sequences.txt
"""

import argparse
import json
import os
from typing import List, Tuple


def generate_task_sequences(
    num_clients, num_tasks_per_client, classes_per_task, total_classes
):
    """
    Generate personalized task sequences for clients in a rotating order.

    Args:
        num_clients (int): Total number of clients
        num_tasks_per_client (int): Number of tasks each client will perform
        classes_per_task (int): Number of classes in each task
        total_classes (int): Total number of classes available in the dataset

    Returns:
        str: Formatted string for -client_seq argument
        dict: Dictionary with client sequences for analysis
    """
    # Validate inputs
    if classes_per_task <= 0 or num_tasks_per_client <= 0 or num_clients <= 0:
        raise ValueError("All parameters must be positive integers")

    if total_classes < classes_per_task:
        raise ValueError(
            f"Total classes ({total_classes}) must be >= classes per task ({classes_per_task})"
        )

    # Warning for potential class overlap
    total_classes_needed = num_clients * classes_per_task
    if total_classes_needed > total_classes:
        print(
            f"[WARNING] With {num_clients} clients and {classes_per_task} classes per task starting position,"
        )
        print(
            f"          we need {total_classes_needed} distinct starting classes but only have {total_classes}."
        )
        print(f"          Classes will wrap around and repeat.")

    all_sequences = []
    client_sequences_dict = {}

    for client_id in range(num_clients):
        # Each client starts with an offset based on their ID
        # Client 0 starts with class 0, Client 1 starts with class classes_per_task, etc.
        start_offset = (client_id * classes_per_task) % total_classes

        client_tasks = []
        current_offset = start_offset

        for task_idx in range(num_tasks_per_client):
            task_classes = []
            for i in range(classes_per_task):
                class_id = (current_offset + i) % total_classes
                task_classes.append(class_id)

            client_tasks.append(task_classes)
            # Move to next consecutive set of classes
            current_offset = (current_offset + classes_per_task) % total_classes

        # Store for analysis
        client_sequences_dict[client_id] = client_tasks

        # Format for -client_seq: "client_id:class1,class2|class3,class4|..."
        tasks_str = "|".join([",".join(map(str, task)) for task in client_tasks])
        all_sequences.append(f"{client_id}:{tasks_str}")

    # Join all client sequences with semicolons
    client_seq_string = ";".join(all_sequences)

    return client_seq_string, client_sequences_dict


def save_sequences(sequences_str, sequences_dict, output_file):
    """Save sequences in multiple formats for convenience."""
    base_name = os.path.splitext(output_file)[0]

    # Save the main format for -client_seq
    with open(output_file, 'w') as f:
        f.write(sequences_str)

    # Save a JSON version for analysis
    json_file = f"{base_name}_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(
            {
                'client_sequences': sequences_dict,
                'client_seq_string': sequences_str,
                'metadata': {
                    'num_clients': len(sequences_dict),
                    'num_tasks_per_client': (
                        len(list(sequences_dict.values())[0]) if sequences_dict else 0
                    ),
                    'classes_per_task': (
                        len(list(sequences_dict.values())[0][0])
                        if sequences_dict and sequences_dict[0]
                        else 0
                    ),
                },
            },
            f,
            indent=2,
        )

    # Save a human-readable version
    readable_file = f"{base_name}_readable.txt"
    with open(readable_file, 'w') as f:
        f.write("=== Generated Task Sequences ===\n\n")
        for client_id, tasks in sequences_dict.items():
            f.write(f"Client {client_id}:\n")
            for task_idx, task_classes in enumerate(tasks):
                f.write(f"  Task {task_idx}: classes {task_classes}\n")
            f.write("\n")
        f.write(f"\nClient Sequence String for -client_seq argument:\n")
        f.write(f'"{sequences_str}"\n')

    return json_file, readable_file


def print_examples(sequences_dict, total_classes):
    """Print example sequences for verification."""
    print("\n=== Example Sequences (first 3 clients, first 5 tasks) ===")
    for client_id in sorted(sequences_dict.keys())[:3]:
        tasks = sequences_dict[client_id][:5]  # First 5 tasks only
        tasks_str = ", ".join([f"({','.join(map(str, task))})" for task in tasks])
        print(f"Client {client_id}: {tasks_str}")
        if len(sequences_dict[client_id]) > 5:
            print(f"          ... and {len(sequences_dict[client_id]) - 5} more tasks")

    if len(sequences_dict) > 3:
        print(f"... and {len(sequences_dict) - 3} more clients")


def main():
    parser = argparse.ArgumentParser(
        description="Generate personalized task sequences for PFTIL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sequences for 5 clients, 10 tasks each, 2 classes per task, Cifar10 (10 classes)
  python generate_task_sequence.py -nc 5 -ntpc 10 -cpt 2 -tc 10 -o cifar10_5c_10t.txt

  # Generate sequences for 50 clients, 50 tasks each, 2 classes per task, Cifar100 (100 classes)
  python generate_task_sequence.py -nc 50 -ntpc 50 -cpt 2 -tc 100 -o cifar100_50c_50t.txt

  # Use in run_apop.sh:
  python main.py ... -client_seq "$(cat cifar10_5c_10t.txt)" ...
        """,
    )

    parser.add_argument(
        '-nc',
        '--num_clients',
        type=int,
        default=5,
        help='Total number of clients (default: 5)',
    )
    parser.add_argument(
        '-ntpc',
        '--num_tasks_per_client',
        type=int,
        default=10,
        help='Number of tasks each client will perform (default: 10)',
    )
    parser.add_argument(
        '-cpt',
        '--classes_per_task',
        type=int,
        default=2,
        help='Number of classes in each task (default: 2)',
    )
    parser.add_argument(
        '-tc',
        '--total_classes',
        type=int,
        default=10,
        help='Total number of classes in the dataset (default: 10 for Cifar10)',
    )
    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        default='client_sequences.txt',
        help='Output file to save sequences (default: client_sequences.txt)',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Show detailed verification of generated sequences',
    )

    args = parser.parse_args()

    print("=== Task Sequence Generator for PFTIL ===")
    print(f"Configuration:")
    print(f"  • Clients: {args.num_clients}")
    print(f"  • Tasks per client: {args.num_tasks_per_client}")
    print(f"  • Classes per task: {args.classes_per_task}")
    print(f"  • Total dataset classes: {args.total_classes}")
    print(f"  • Output file: {args.output_file}")

    try:
        # Generate sequences
        sequences_str, sequences_dict = generate_task_sequences(
            args.num_clients,
            args.num_tasks_per_client,
            args.classes_per_task,
            args.total_classes,
        )

        # Save in multiple formats
        json_file, readable_file = save_sequences(
            sequences_str, sequences_dict, args.output_file
        )

        print(f"\n=== Files Generated ===")
        print(f"  • Main sequence file: {os.path.abspath(args.output_file)}")
        print(f"  • Analysis JSON: {os.path.abspath(json_file)}")
        print(f"  • Human-readable: {os.path.abspath(readable_file)}")

        # Show examples
        print_examples(sequences_dict, args.total_classes)

        print(f"\n=== Usage in run_apop.sh ===")
        print(f'Add this to your run_apop.sh script:')
        print(f'-client_seq "$(cat {args.output_file})"')

        # Verification mode
        if args.verify:
            print(f"\n=== Verification ===")
            # Check for class distribution
            all_classes = set()
            for client_tasks in sequences_dict.values():
                for task in client_tasks:
                    all_classes.update(task)
            print(f"Classes used: {sorted(all_classes)} (total: {len(all_classes)})")

            # Check task overlap between clients
            print(f"First task of each client:")
            for i in range(min(5, args.num_clients)):
                first_task = sequences_dict[i][0]
                print(f"  Client {i}: {first_task}")

        print(f"\n✓ Task sequences generated successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
