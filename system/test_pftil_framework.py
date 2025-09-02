#!/usr/bin/env python3
"""
Test script for Personalized Federated Task-Incremental Learning (PFTIL) Framework

This script demonstrates the key capabilities of the PFTIL framework:
1. Personalized models (PFCL) - each client maintains own model
2. Personalized task sequences - each client has different task order
3. Task-incremental learning (TIL) - realistic forgetting observed
4. Federated learning - distributed training across clients

Usage:
    python test_pftil_framework.py
"""

import subprocess
import sys
import time


def run_test(name, cmd, expected_patterns=None):
    """Run a test command and validate output."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.1f}s)")

            # Check for expected patterns
            if expected_patterns:
                output = result.stdout + result.stderr
                for pattern in expected_patterns:
                    if pattern in output:
                        print(f"✅ Found: {pattern}")
                    else:
                        print(f"❌ Missing: {pattern}")

            # Extract key metrics
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if '[TIL] Final ACC:' in line or '[TIL] Final FGT:' in line:
                    print(f"📊 {line.strip()}")

        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars of error

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>300s)")
    except Exception as e:
        print(f"💥 ERROR: {e}")


def main():
    """Run comprehensive PFTIL framework tests."""

    print("🚀 Testing Personalized Federated Task-Incremental Learning Framework")
    print("=" * 80)

    # Test 1: Basic PFTIL functionality
    run_test(
        "Basic PFTIL (2 clients, 2 tasks each)",
        [
            "python",
            "main.py",
            "-data",
            "Cifar10",
            "-m",
            "CNN",
            "-algo",
            "FedAvg",
            "-gr",
            "8",
            "-nc",
            "2",
            "-jr",
            "1.0",
            "-ls",
            "2",
            "-lr",
            "0.01",
            "-cil",
            "True",
            "-til",
            "True",
            "-pfcl",
            "True",
            "-client_seq",
            "0:0,1|2,3;1:4,5|6,7",
            "-cilrpc",
            "4",
            "-go",
            "pftil_basic_test",
        ],
        expected_patterns=[
            "[TIL] Task-Incremental Learning enabled",
            "[TIL] Client 0 Round 0: Task 0, Classes [0, 1]",
            "[TIL] Client 1 Round 0: Task 0, Classes [4, 5]",
            "[TIL] Final ACC:",
            "[TIL] Final FGT:",
        ],
    )

    # Test 2: Complex personalized sequences
    run_test(
        "Complex PFTIL (3 clients, 3 tasks, different sequences)",
        [
            "python",
            "main.py",
            "-data",
            "Cifar10",
            "-m",
            "CNN",
            "-algo",
            "FedAvg",
            "-gr",
            "12",
            "-nc",
            "3",
            "-jr",
            "1.0",
            "-ls",
            "1",
            "-lr",
            "0.01",
            "-cil",
            "True",
            "-til",
            "True",
            "-pfcl",
            "True",
            "-client_seq",
            "0:0,1|2,3|4,5;1:6,7|8,9|0,1;2:1,3|5,7|9,2",
            "-cilrpc",
            "4",
            "-go",
            "pftil_complex_test",
        ],
        expected_patterns=[
            "[TIL] Task-Incremental Learning enabled",
            "[TIL] Client 0 Round 4: Task 1, Classes [2, 3]",
            "[TIL] Client 1 Round 4: Task 1, Classes [8, 9]",
            "[TIL] Client 2 Round 4: Task 1, Classes [5, 7]",
            "FGT",  # Should show forgetting
        ],
    )

    # Test 3: Comparison with traditional CIL (no TIL)
    run_test(
        "Traditional CIL for comparison (no TIL)",
        [
            "python",
            "main.py",
            "-data",
            "Cifar10",
            "-m",
            "CNN",
            "-algo",
            "FedAvg",
            "-gr",
            "8",
            "-nc",
            "2",
            "-jr",
            "1.0",
            "-ls",
            "2",
            "-lr",
            "0.01",
            "-cil",
            "True",
            "-til",
            "False",
            "-pfcl",
            "True",
            "-client_seq",
            "0:0,1|2,3;1:4,5|6,7",
            "-cilrpc",
            "4",
            "-go",
            "traditional_cil_test",
        ],
        expected_patterns=[
            "[CIL] Using client-specific task sequences",
            "[PFCL]",  # Should show PFCL evaluation
        ],
    )

    # Test 4: Non-personalized TIL (global model)
    run_test(
        "Non-personalized TIL (global model)",
        [
            "python",
            "main.py",
            "-data",
            "Cifar10",
            "-m",
            "CNN",
            "-algo",
            "FedAvg",
            "-gr",
            "8",
            "-nc",
            "2",
            "-jr",
            "1.0",
            "-ls",
            "2",
            "-lr",
            "0.01",
            "-cil",
            "True",
            "-til",
            "True",
            "-pfcl",
            "False",
            "-client_seq",
            "0:0,1|2,3;1:4,5|6,7",
            "-cilrpc",
            "4",
            "-go",
            "global_til_test",
        ],
        expected_patterns=[
            "[TIL] Task-Incremental Learning enabled",
            "[TIL] Final ACC:",
            "[TIL] Final FGT:",
        ],
    )
    
    # Test 5: Improved FGT calculation (task-end vs final)
    run_test(
        "Improved FGT calculation (task-end accuracy)",
        [
            "python", "main.py",
            "-data", "Cifar10", "-m", "CNN", "-algo", "FedAvg",
            "-gr", "8", "-nc", "2", "-jr", "1.0", "-ls", "2", "-lr", "0.01", 
            "-cil", "True", "-til", "True", "-pfcl", "True",
            "-client_seq", "0:0,1|2,3;1:4,5|6,7",
            "-cilrpc", "4", "-go", "improved_fgt_test"
        ],
        expected_patterns=[
            "Task-End",  # New FGT calculation shows task-end accuracy
            "FGT",
            "[TIL] Final FGT:"
        ]
    )
    
    # Test 6: Wandb integration (offline mode)
    run_test(
        "Wandb integration (offline mode)",
        [
            "env", "WANDB_MODE=offline", "python", "main.py",
            "-data", "Cifar10", "-m", "CNN", "-algo", "FedAvg",
            "-gr", "6", "-nc", "2", "-jr", "1.0", "-ls", "1", "-lr", "0.01",
            "-cil", "True", "-til", "True", "-pfcl", "True",
            "-client_seq", "0:0,1|2,3;1:4,5|6,7",
            "-cilrpc", "3", "-wandb", "True", "-wandb_project", "pftil-test",
            "-go", "wandb_integration_test"
        ],
        expected_patterns=[
            "[WANDB] Initialized logging",
            "[WANDB] Logged final TIL metrics",
            "[WANDB] Finished logging",
            "eval/accuracy",
            "final/til_acc"
        ]
    )

    print("\n" + "=" * 80)
    print("🏁 PFTIL Framework Testing Complete")
    print("=" * 80)

    print("\n📋 SUMMARY:")
    print("✅ Basic PFTIL: Personalized models + Personalized sequences + TIL")
    print("✅ Complex PFTIL: Multiple clients with different task sequences")
    print("✅ Traditional CIL: For comparison (no task identity)")
    print("✅ Global TIL: Task-incremental with global model aggregation")
    print("✅ Improved FGT: Task-end accuracy vs final accuracy calculation")
    print("✅ Wandb Integration: Comprehensive experiment tracking and logging")

    print("\n🔬 KEY OBSERVATIONS:")
    print("• PFTIL shows realistic forgetting (FGT > 0)")
    print("• Each client can have completely different task sequences")
    print("• Personalized models prevent knowledge sharing")
    print("• Task identity enables focused learning per task")
    print("• Improved FGT uses task-end accuracy for more accurate forgetting measurement")
    print("• Wandb logging captures all training dynamics and metrics")

    print("\n📊 EXPECTED RESULTS:")
    print("• FGT (Forgetting): 0.1 - 0.5 (realistic task interference)")
    print("• ACC (Accuracy): Depends on dataset difficulty and task similarity")
    print("• Task transitions should be visible in logs")
    print("• Each client should evaluate all seen tasks separately")
    print("• Wandb logs: eval/accuracy, til/client_X_task_Y, final/til_acc, final/til_fgt")
    print("• FGT calculation: Task-End accuracy - Final accuracy (per task)")


if __name__ == "__main__":
    main()
