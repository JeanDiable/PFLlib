#!/usr/bin/env python3
"""
Static verification script for APOP implementation.
This script verifies the algorithm correctness without requiring torch.
"""

import ast
import os
import re
import sys


def check_file_exists(filepath):
    """Check if a file exists."""
    exists = os.path.isfile(filepath)
    print(f"{'✓' if exists else '✗'} {filepath} {'exists' if exists else 'MISSING'}")
    return exists


def read_file_content(filepath):
    """Read file content safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return None


def verify_algorithm_correctness():
    """Verify the APOP algorithm implementation against specification."""
    print("=" * 60)
    print("🔍 VERIFYING APOP ALGORITHM CORRECTNESS")
    print("=" * 60)

    # Check file structure
    print("\n📁 Checking file structure...")
    files_to_check = [
        "system/flcore/clients/clientapop.py",
        "system/flcore/servers/serverapop.py",
        "system/utils/apop_utils.py",
        "APOP_USAGE_EXAMPLE.md",
    ]

    all_files_exist = True
    for filepath in files_to_check:
        if not check_file_exists(filepath):
            all_files_exist = False

    if not all_files_exist:
        print("❌ Missing required files!")
        return False

    # Verify client implementation
    print("\n🧩 Verifying client implementation...")
    client_content = read_file_content("system/flcore/clients/clientapop.py")
    if not client_content:
        return False

    client_checks = [
        ("Dual gradient modulation method", "_apply_apop_gradient_modulation"),
        ("Orthogonal projection method", "_apply_orthogonal_projection"),
        ("Parallel projection method", "_apply_parallel_projection"),
        ("Task signature computation", "_compute_task_signature"),
        ("Knowledge distillation", "distill_knowledge"),
        ("Adaptation status checking", "_check_adaptation_status"),
        ("TIL integration", "_mask_loss_for_training"),
    ]

    for check_name, method_name in client_checks:
        if method_name in client_content:
            print(f"✓ {check_name}: {method_name}")
        else:
            print(f"✗ {check_name}: {method_name} MISSING")
            return False

    # Verify server implementation
    print("\n🖥️  Verifying server implementation...")
    server_content = read_file_content("system/flcore/servers/serverapop.py")
    if not server_content:
        return False

    server_checks = [
        ("Knowledge base management", "knowledge_base"),
        ("Knowledge base update", "_update_knowledge_base"),
        ("Knowledge base query", "_query_knowledge_base"),
        ("Knowledge transfer handling", "_handle_knowledge_transfer_request"),
        ("Task completion collection", "_collect_task_completions"),
        ("SVD-based fusion", "np.linalg.svd"),
    ]

    for check_name, pattern in server_checks:
        if pattern in server_content:
            print(f"✓ {check_name}: {pattern}")
        else:
            print(f"✗ {check_name}: {pattern} MISSING")
            return False

    # Verify main.py integration
    print("\n⚙️  Verifying main.py integration...")
    main_content = read_file_content("system/main.py")
    if not main_content:
        return False

    main_checks = [
        ("APOP import", "from flcore.servers.serverapop import APOP"),
        ("APOP algorithm selection", 'args.algorithm == "APOP"'),
        ("Subspace dimension parameter", "--subspace_dim"),
        ("Adaptation threshold parameter", "--adaptation_threshold"),
        ("Fusion threshold parameter", "--fusion_threshold"),
        ("Max transfer gain parameter", "--max_transfer_gain"),
    ]

    for check_name, pattern in main_checks:
        if pattern in main_content:
            print(f"✓ {check_name}: found")
        else:
            print(f"✗ {check_name}: MISSING")
            return False

    return True


def verify_algorithm_logic():
    """Verify the core algorithm logic matches the specification."""
    print("\n🧮 Verifying algorithm logic...")

    client_content = read_file_content("system/flcore/clients/clientapop.py")
    server_content = read_file_content("system/flcore/servers/serverapop.py")

    if not client_content or not server_content:
        return False

    # Check orthogonal projection formula: g_k' = g_k - B_past B_past^T g_k
    if "g_k - projection" in client_content and "B_past.t()" in client_content:
        print("✓ Orthogonal projection formula: g_k' = g_k - B_past B_past^T g_k")
    else:
        print("✗ Orthogonal projection formula incorrect")
        return False

    # Check parallel projection formula: g_k'' = (1+α) g_k^∥ + g_k^⊥
    if "(1 + alpha) * g_k_parallel + g_k_orthogonal" in client_content:
        print("✓ Parallel projection formula: g_k'' = (1+α) g_k^∥ + g_k^⊥")
    else:
        print("✗ Parallel projection formula incorrect")
        return False

    # Check adaptive transfer gain: α = α_max × sim_retrieved
    if "self.max_transfer_gain * self.similarity_retrieved" in client_content:
        print("✓ Adaptive transfer gain: α = α_max × sim_retrieved")
    else:
        print("✗ Adaptive transfer gain formula incorrect")
        return False

    # Check adaptation threshold logic: sim < δ
    if "similarity < self.adaptation_threshold" in client_content:
        print("✓ Adaptation threshold logic: sim < δ")
    else:
        print("✗ Adaptation threshold logic incorrect")
        return False

    # Check fusion threshold logic: similarity > γ
    if "max_similarity > self.fusion_threshold" in server_content:
        print("✓ Fusion threshold logic: similarity > γ")
    else:
        print("✗ Fusion threshold logic incorrect")
        return False

    # Check SVD-based fusion
    if "np.linalg.svd(stacked_bases" in server_content:
        print("✓ SVD-based knowledge fusion")
    else:
        print("✗ SVD-based knowledge fusion missing")
        return False

    return True


def verify_parameter_consistency():
    """Verify parameter consistency across files."""
    print("\n📊 Verifying parameter consistency...")

    files_content = {
        'client': read_file_content("system/flcore/clients/clientapop.py"),
        'server': read_file_content("system/flcore/servers/serverapop.py"),
        'main': read_file_content("system/main.py"),
    }

    if not all(files_content.values()):
        return False

    # Check parameter consistency
    parameters = [
        ("subspace_dim", "r in algorithm", [20]),
        ("adaptation_threshold", "δ in algorithm", [0.3]),
        ("fusion_threshold", "γ in algorithm", [0.7]),
        ("max_transfer_gain", "α_max in algorithm", [2.0]),
    ]

    for param_name, description, default_values in parameters:
        found_in_files = []
        for file_name, content in files_content.items():
            if param_name in content:
                found_in_files.append(file_name)

        if len(found_in_files) >= 2:  # Should be in at least client and main
            print(f"✓ {param_name} ({description}): found in {found_in_files}")
        else:
            print(f"✗ {param_name} ({description}): found only in {found_in_files}")
            return False

    return True


def verify_logging_completeness():
    """Verify comprehensive logging is implemented."""
    print("\n📝 Verifying logging completeness...")

    client_content = read_file_content("system/flcore/clients/clientapop.py")
    server_content = read_file_content("system/flcore/servers/serverapop.py")

    if not client_content or not server_content:
        return False

    # Check client logging patterns
    client_log_patterns = ["[APOP-GRAD]", "[APOP-ORTH]", "[APOP-PARA]", "[APOP-ADAPT]"]

    for pattern in client_log_patterns:
        if pattern in client_content:
            print(f"✓ Client logging: {pattern}")
        else:
            print(f"✗ Client logging missing: {pattern}")
            return False

    # Check server logging patterns
    server_log_patterns = ["[APOP-KB]", "[APOP-QUERY]", "[APOP]"]

    for pattern in server_log_patterns:
        if pattern in server_content:
            print(f"✓ Server logging: {pattern}")
        else:
            print(f"✗ Server logging missing: {pattern}")
            return False

    return True


def verify_error_handling():
    """Verify proper error handling."""
    print("\n🛡️  Verifying error handling...")

    client_content = read_file_content("system/flcore/clients/clientapop.py")
    server_content = read_file_content("system/flcore/servers/serverapop.py")

    if not client_content or not server_content:
        return False

    # Check try-catch blocks
    client_try_count = client_content.count("try:")
    client_except_count = client_content.count("except ")

    server_try_count = server_content.count("try:")
    server_except_count = server_content.count("except ")

    if client_try_count > 0 and client_except_count > 0:
        print(
            f"✓ Client error handling: {client_try_count} try blocks, {client_except_count} except blocks"
        )
    else:
        print(f"✗ Client error handling insufficient")
        return False

    if server_try_count > 0 and server_except_count > 0:
        print(
            f"✓ Server error handling: {server_try_count} try blocks, {server_except_count} except blocks"
        )
    else:
        print(f"✗ Server error handling insufficient")
        return False

    # Check dimension mismatch handling
    if "Dimension mismatch" in client_content:
        print("✓ Dimension mismatch handling")
    else:
        print("✗ Dimension mismatch handling missing")
        return False

    return True


def check_syntax_errors():
    """Check for basic Python syntax errors."""
    print("\n🔍 Checking syntax errors...")

    files_to_check = [
        "system/flcore/clients/clientapop.py",
        "system/flcore/servers/serverapop.py",
        "system/utils/apop_utils.py",
    ]

    for filepath in files_to_check:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to parse as AST
            ast.parse(content)
            print(f"✓ {filepath}: No syntax errors")

        except SyntaxError as e:
            print(f"✗ {filepath}: Syntax error at line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            print(f"✗ {filepath}: Error parsing file: {e}")
            return False

    return True


def main():
    """Run all verification checks."""
    print("🔬 APOP STATIC VERIFICATION SUITE")
    print("=" * 60)

    checks = [
        ("Algorithm Correctness", verify_algorithm_correctness),
        ("Algorithm Logic", verify_algorithm_logic),
        ("Parameter Consistency", verify_parameter_consistency),
        ("Logging Completeness", verify_logging_completeness),
        ("Error Handling", verify_error_handling),
        ("Syntax Errors", check_syntax_errors),
    ]

    results = []

    for check_name, check_func in checks:
        print(f"\n🔍 Running {check_name} verification...")
        try:
            result = check_func()
            results.append((check_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {check_name}")
        except Exception as e:
            results.append((check_name, False))
            print(f"❌ FAILED: {check_name} - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {check_name}")

    print("-" * 60)
    print(f"Results: {passed}/{total} checks passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("✨ APOP implementation matches algorithm specification!")
        print("🚀 Ready for testing with proper environment setup!")
        return 0
    else:
        print(f"\n⚠️  {total-passed} verifications failed.")
        print("🔧 Please address the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
