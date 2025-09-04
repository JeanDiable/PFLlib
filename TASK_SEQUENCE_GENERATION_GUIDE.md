# Task Sequence Generation Guide for PFTIL

This guide explains how to generate personalized task sequences for the PFTIL (Personalized Federated Task-Incremental Learning) setting using the `generate_task_sequence.py` script.

## Key Parameter Relationships

**IMPORTANT**: Understanding the relationship between parameters:

- `cilrpc` (cil_rounds_per_class) = **Rounds per TASK** (despite the name)
- Total rounds = `num_tasks_per_client × cilrpc`  
- `cilrpc` is the **main parameter you adjust** based on your experimental needs
- Task structure is determined by `num_tasks_per_client` and `classes_per_task`

## Overview

The task sequence generator creates personalized task sequences following a **rotating order pattern**:
- Client 0: (0,1), (2,3), (4,5), (6,7), ...
- Client 1: (2,3), (4,5), (6,7), (8,9), ...
- Client 2: (4,5), (6,7), (8,9), (10,11), ...

This ensures each client starts with different classes but follows a consistent progression pattern.

## Quick Start

### Basic Usage
```bash
# Generate sequences for 5 clients, 10 tasks each, 2 classes per task, Cifar10
python generate_task_sequence.py -nc 5 -ntpc 10 -cpt 2 -tc 10 -o sequences.txt

# Use in run_apop.sh
./run_apop.sh sequences.txt Cifar10 5
```

### Auto-generation (recommended)
```bash
# Auto-generate sequences for Cifar100 with 10 clients
./run_apop.sh "" Cifar100 10

# Auto-generate sequences for Cifar10 with 5 clients
./run_apop.sh "" Cifar10 5
```

## Parameters

| Parameter                | Short   | Description                | Default                |
| ------------------------ | ------- | -------------------------- | ---------------------- |
| `--num_clients`          | `-nc`   | Number of clients          | 5                      |
| `--num_tasks_per_client` | `-ntpc` | Tasks per client           | 10                     |
| `--classes_per_task`     | `-cpt`  | Classes per task           | 2                      |
| `--total_classes`        | `-tc`   | Total dataset classes      | 10                     |
| `--output_file`          | `-o`    | Output file path           | `client_sequences.txt` |
| `--verify`               |         | Show detailed verification | False                  |

## Examples

### Example 1: Small-scale (Cifar10)
```bash
# 5 clients, 5 tasks each, 2 classes per task
python generate_task_sequence.py -nc 5 -ntpc 5 -cpt 2 -tc 10 -o cifar10_small.txt
```

**Generated sequences:**
- Client 0: (0,1), (2,3), (4,5), (6,7), (8,9)
- Client 1: (2,3), (4,5), (6,7), (8,9), (0,1)
- Client 2: (4,5), (6,7), (8,9), (0,1), (2,3)
- Client 3: (6,7), (8,9), (0,1), (2,3), (4,5)
- Client 4: (8,9), (0,1), (2,3), (4,5), (6,7)

### Example 2: Medium-scale (Cifar100)
```bash
# 10 clients, 20 tasks each, 2 classes per task
python generate_task_sequence.py -nc 10 -ntpc 20 -cpt 2 -tc 100 -o cifar100_medium.txt
```

### Example 3: Large-scale
```bash
# 50 clients, 50 tasks each, 2 classes per task, Cifar100
python generate_task_sequence.py -nc 50 -ntpc 50 -cpt 2 -tc 100 -o cifar100_large.txt
```

### Example 4: Different task sizes
```bash
# 5 clients, 10 tasks each, 5 classes per task (larger tasks)
python generate_task_sequence.py -nc 5 -ntpc 10 -cpt 5 -tc 100 -o large_tasks.txt
```

## Output Files

The script generates three files for each run:

### 1. Main Sequence File (`.txt`)
Contains the formatted string for the `-client_seq` argument:
```
0:0,1|2,3|4,5;1:2,3|4,5|6,7;2:4,5|6,7|8,9
```

### 2. Human-readable File (`_readable.txt`)
Shows the sequences in a readable format:
```
=== Generated Task Sequences ===

Client 0:
  Task 0: classes [0, 1]
  Task 1: classes [2, 3]
  Task 2: classes [4, 5]

Client 1:
  Task 0: classes [2, 3]
  Task 1: classes [4, 5]
  Task 2: classes [6, 7]
```

### 3. Analysis File (`_analysis.json`)
Contains structured data for analysis:
```json
{
  "client_sequences": {...},
  "client_seq_string": "...",
  "metadata": {
    "num_clients": 5,
    "num_tasks_per_client": 10,
    "classes_per_task": 2
  }
}
```

## Integration with run_apop.sh

The script now correctly calculates total rounds as: **Total Rounds = num_tasks_per_client × cilrpc**

Where:
- `cilrpc` = rounds per **task** (not per class) - this is the main parameter you adjust
- `num_tasks_per_client` = number of tasks each client performs
- `classes_per_task` = number of classes in each task

### Method 1: Auto-generation with custom cilrpc (recommended)
```bash
# Auto-generate sequences with 15 rounds per task
./run_apop.sh "" Cifar100 10 15

# Auto-generate sequences with 5 rounds per task for quick testing  
./run_apop.sh "" Cifar10 5 5
```

### Method 2: Pre-generated sequences
```bash
# Generate sequences first
python generate_task_sequence.py -nc 10 -ntpc 20 -cpt 5 -tc 100 -o my_sequences.txt

# Use in experiment with 12 rounds per task
./run_apop.sh my_sequences.txt Cifar100 10 12
```

### Method 3: Manual specification in script
```bash
cd system
# Example: 20 tasks × 10 rounds per task = 200 total rounds
python main.py ... -client_seq "$(cat ../sequences.txt)" -cilrpc 10 -gr 200 ...
```

### Example Calculations:
- 20 tasks per client × 5 rounds per task = 100 total rounds
- 10 tasks per client × 15 rounds per task = 150 total rounds  
- 50 tasks per client × 4 rounds per task = 200 total rounds

## Advanced Usage

### Verification Mode
```bash
python generate_task_sequence.py -nc 5 -ntpc 10 -cpt 2 -tc 20 --verify
```
Shows detailed information about class distribution and overlaps.

### Custom Dataset Configurations
```bash
# For a custom dataset with 50 classes
python generate_task_sequence.py -nc 20 -ntpc 25 -cpt 2 -tc 50 -o custom.txt

# For larger tasks (5 classes each)
python generate_task_sequence.py -nc 10 -ntpc 10 -cpt 5 -tc 100 -o large_tasks.txt
```

## Tips and Best Practices

1. **Class Distribution**: Ensure `total_classes >= classes_per_task` for meaningful sequences
2. **Overlap Handling**: If `num_clients * classes_per_task > total_classes`, classes will wrap around
3. **File Naming**: Use descriptive names like `cifar100_50c_50t.txt` for easy identification
4. **Verification**: Always use `--verify` for new configurations to check class distribution
5. **Automation**: Use `run_apop.sh` auto-generation for standard experiments

## Troubleshooting

### Common Issues

**Issue**: "Classes will wrap around and repeat"
- **Cause**: Too many clients relative to available classes
- **Solution**: Increase `total_classes` or reduce `num_clients`/`classes_per_task`

**Issue**: Empty or malformed sequences
- **Cause**: Invalid parameter combinations
- **Solution**: Verify all parameters are positive integers

**Issue**: Experiment fails with sequence parsing error
- **Cause**: Corrupted sequence file
- **Solution**: Regenerate the sequence file

### Getting Help
```bash
python generate_task_sequence.py --help
./run_apop.sh
```

## Implementation Details

The script uses a rotating offset algorithm:
1. Each client starts at offset `client_id * classes_per_task`
2. Each task advances by `classes_per_task` 
3. Class indices wrap around using modulo `total_classes`

This ensures:
- **Diversity**: Each client starts with different classes
- **Fairness**: All clients eventually see all classes (given enough tasks)
- **Consistency**: Deterministic and reproducible sequences
- **Overlap Control**: Gradual introduction of shared classes between clients
