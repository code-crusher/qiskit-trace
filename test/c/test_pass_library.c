// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// This tests the built-in transpiler pass library exposed through the pass-manager C API; see
// `qk_pass_builtin_new` and friends (issue #17251).

#include "common.h"
#include <inttypes.h>
#include <math.h>
#include <qiskit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED_VARIABLE(x) (void)(x)

/// Build a pass manager holding the given built-in pass, ready to run on the DAG IR.
///
/// Returns `NULL` and sets `result` to `RuntimeError` if any of the construction steps fail.
static QkPassManager *pm_with_builtin(uint32_t builtin, QkIrHandle *dag_ir, int *result) {
    UNUSED_VARIABLE(dag_ir);
    QkPass *pass = qk_pass_builtin_new(builtin);
    if (pass == NULL) {
        fprintf(stderr, "Failed to create builtin pass %" PRIu32 ".\n", builtin);
        *result = RuntimeError;
        return NULL;
    }
    QkPassManager *pm = qk_passmanager_new();
    if (pm == NULL) {
        fprintf(stderr, "Failed to create pass manager.\n");
        qk_pass_free(pass);
        *result = RuntimeError;
        return NULL;
    }
    if (qk_passmanager_push_pass(pm, pass) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing builtin pass.\n");
        qk_passmanager_free(pm);
        qk_pass_free(pass);
        *result = RuntimeError;
        return NULL;
    }
    // Ownership of the pass was taken by the pass manager; only the PM needs freeing.
    return pm;
}

/// Assert that the DAG contains exactly the named operations, in order.
static int dag_op_names_equal(const QkDag *dag, const char *const *expected, size_t expected_len) {
    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != expected_len) {
        fprintf(stderr, "Expected %zu operations, but got %zu.\n", expected_len, num_ops);
        return EqualityError;
    }
    uint32_t *op_indices = malloc(num_ops * sizeof(*op_indices));
    qk_dag_topological_op_nodes((QkDag *)dag, op_indices);
    int result = Ok;
    for (size_t i = 0; i < num_ops; i++) {
        QkCircuitInstruction inst;
        qk_dag_get_instruction((QkDag *)dag, op_indices[i], &inst);
        if (strcmp(inst.name, expected[i]) != 0) {
            fprintf(stderr, "Expected %s at %zu, but got %s\n", expected[i], i, inst.name);
            result = EqualityError;
        }
        qk_circuit_instruction_clear(&inst);
        if (result != Ok) {
            break;
        }
    }
    free(op_indices);
    return result;
}

static int test_builtin_pass_invalid_id(void) {
    int result = Ok;
    if (qk_pass_builtin_new(3) != NULL) {
        fprintf(stderr, "Expected NULL for an invalid builtin id.\n");
        result = EqualityError;
    }
    if (qk_pass_builtin_new(UINT32_MAX) != NULL) {
        fprintf(stderr, "Expected NULL for UINT32_MAX as builtin id.\n");
        result = EqualityError;
    }
    return result;
}

static int test_builtin_remove_identity_equivalent(void) {
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);
    int result = Ok;
    QkPassManager *pm = pm_with_builtin(QkPassBuiltin_RemoveIdentityEquivalent, dag_ir, &result);
    if (pm == NULL) {
        goto cleanup_ir;
    }

    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    double nonzero[1] = {1.23};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_RX, q0, almost_zero);
    qk_circuit_gate(circuit, QkGate_RZ, q0, nonzero);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);

    QkCompilationError *error = NULL;
    QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, (void *)dag, dag_ir, dag_ir, &error);
    if (error != NULL || out == NULL) {
        fprintf(stderr, "Failed running builtin RemoveIdentityEquivalent.\n");
        if (error != NULL) {
            qk_compilation_error_free(error);
        } else {
            qk_dag_free(dag);
        }
        result = RuntimeError;
        goto cleanup_pm;
    }

    // `rx(1e-20)` is equivalent to the identity and must be removed; the other gates survive.
    const char *const expected[] = {"h", "rz", "h"};
    int cmp = dag_op_names_equal(out, expected, 3);
    if (cmp != Ok) {
        result = cmp;
    }
    qk_dag_free(out);

cleanup_pm:
    qk_passmanager_free(pm);
cleanup_ir:
    qk_ir_handle_free(dag_ir);
    return result;
}

static int test_builtin_remove_diagonal_gates_before_measure(void) {
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);
    int result = Ok;
    QkPassManager *pm =
        pm_with_builtin(QkPassBuiltin_RemoveDiagonalGatesBeforeMeasure, dag_ir, &result);
    if (pm == NULL) {
        goto cleanup_ir;
    }

    QkCircuit *circuit = qk_circuit_new(1, 1);
    uint32_t q0[1] = {0};
    double theta[1] = {1.23};
    qk_circuit_gate(circuit, QkGate_X, q0, NULL);
    qk_circuit_gate(circuit, QkGate_RZ, q0, theta);
    qk_circuit_measure(circuit, 0, 0);
    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);

    QkCompilationError *error = NULL;
    QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, (void *)dag, dag_ir, dag_ir, &error);
    if (error != NULL || out == NULL) {
        fprintf(stderr, "Failed running builtin RemoveDiagonalGatesBeforeMeasure.\n");
        if (error != NULL) {
            qk_compilation_error_free(error);
        } else {
            qk_dag_free(dag);
        }
        result = RuntimeError;
        goto cleanup_pm;
    }

    // The diagonal `rz` directly before the measure is removed; the `x` survives.
    const char *const expected[] = {"x", "measure"};
    int cmp = dag_op_names_equal(out, expected, 2);
    if (cmp != Ok) {
        result = cmp;
    }
    qk_dag_free(out);

cleanup_pm:
    qk_passmanager_free(pm);
cleanup_ir:
    qk_ir_handle_free(dag_ir);
    return result;
}

static int test_builtin_optimize_1q_sequences(void) {
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);
    int result = Ok;
    QkPassManager *pm = pm_with_builtin(QkPassBuiltin_Optimize1qSequences, dag_ir, &result);
    if (pm == NULL) {
        goto cleanup_ir;
    }

    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t q0[1] = {0};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);

    QkCompilationError *error = NULL;
    QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, (void *)dag, dag_ir, dag_ir, &error);
    if (error != NULL || out == NULL) {
        fprintf(stderr, "Failed running builtin Optimize1qGatesDecomposition.\n");
        if (error != NULL) {
            qk_compilation_error_free(error);
        } else {
            qk_dag_free(dag);
        }
        result = RuntimeError;
        goto cleanup_pm;
    }

    // Three Hadamards collapse into a single one-qubit gate.
    size_t num_ops = qk_dag_num_op_nodes(out);
    if (num_ops != 1) {
        fprintf(stderr, "Expected 1 operation after optimization, but got %zu.\n", num_ops);
        result = EqualityError;
    }
    qk_dag_free(out);

cleanup_pm:
    qk_passmanager_free(pm);
cleanup_ir:
    qk_ir_handle_free(dag_ir);
    return result;
}

static int test_builtin_setters_reject_invalid_usage(void) {
    int result = Ok;
    QkPass *diagonal = qk_pass_builtin_new(QkPassBuiltin_RemoveDiagonalGatesBeforeMeasure);
    if (diagonal == NULL) {
        fprintf(stderr, "Failed to create RemoveDiagonalGatesBeforeMeasure pass.\n");
        return RuntimeError;
    }
    QkPass *identity = qk_pass_builtin_new(QkPassBuiltin_RemoveIdentityEquivalent);
    if (identity == NULL) {
        fprintf(stderr, "Failed to create RemoveIdentityEquivalent pass.\n");
        qk_pass_free(diagonal);
        return RuntimeError;
    }

    // The pass built from an invalid id is a null pointer, and the setters must reject it.
    if (qk_pass_builtin_set_target(NULL, NULL) != QkExitCode_InvalidPassOption) {
        fprintf(stderr, "Expected QkExitCode_InvalidPassOption for a null pass.\n");
        result = EqualityError;
    }
    // The diagonal-gates pass has no target or approximation-degree options.
    if (qk_pass_builtin_set_target(diagonal, NULL) != QkExitCode_InvalidPassOption) {
        fprintf(stderr, "Expected QkExitCode_InvalidPassOption for target on a pass without it.\n");
        result = EqualityError;
    }
    if (qk_pass_builtin_set_approximation_degree(diagonal, 0.5) !=
        QkExitCode_InvalidPassOption) {
        fprintf(stderr,
                "Expected QkExitCode_InvalidPassOption for approximation degree on a pass "
                "without it.\n");
        result = EqualityError;
    }
    // The identity-removal pass accepts approximation degrees in [0, 1] and NaN.
    if (qk_pass_builtin_set_approximation_degree(identity, 0.5) != QkExitCode_Success) {
        fprintf(stderr, "Expected success setting a valid approximation degree.\n");
        result = EqualityError;
    }
    if (qk_pass_builtin_set_approximation_degree(identity, NAN) != QkExitCode_Success) {
        fprintf(stderr, "Expected success setting a NaN approximation degree.\n");
        result = EqualityError;
    }
    if (qk_pass_builtin_set_approximation_degree(identity, 1.5) != QkExitCode_InvalidPassOption) {
        fprintf(stderr, "Expected QkExitCode_InvalidPassOption for an out-of-range degree.\n");
        result = EqualityError;
    }
    // Clearing the target is always allowed on passes that accept one.
    if (qk_pass_builtin_set_target(identity, NULL) != QkExitCode_Success) {
        fprintf(stderr, "Expected success clearing the target.\n");
        result = EqualityError;
    }

    qk_pass_free(identity);
    qk_pass_free(diagonal);
    return result;
}

static int test_builtin_set_target_accepts_target(void) {
    int result = Ok;
    QkPass *identity = qk_pass_builtin_new(QkPassBuiltin_RemoveIdentityEquivalent);
    if (identity == NULL) {
        fprintf(stderr, "Failed to create RemoveIdentityEquivalent pass.\n");
        return RuntimeError;
    }
    QkTarget *target = qk_target_new(1);
    if (qk_pass_builtin_set_target(identity, target) != QkExitCode_Success) {
        fprintf(stderr, "Expected success setting a target.\n");
        result = EqualityError;
    }
    // The pass takes its own copy; the caller's target can be released immediately.
    qk_target_free(target);

    // The pass must still be runnable after configuration.
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);
    QkPassManager *pm = qk_passmanager_new();
    if (qk_passmanager_push_pass(pm, identity) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing configured builtin pass.\n");
        result = RuntimeError;
        goto cleanup_pm;
    }

    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    qk_circuit_gate(circuit, QkGate_RX, q0, almost_zero);
    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);

    QkCompilationError *error = NULL;
    QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, (void *)dag, dag_ir, dag_ir, &error);
    if (error != NULL || out == NULL) {
        fprintf(stderr, "Failed running configured builtin pass.\n");
        if (error != NULL) {
            qk_compilation_error_free(error);
        } else {
            qk_dag_free(dag);
        }
        result = RuntimeError;
        goto cleanup_pm;
    }
    if (qk_dag_num_op_nodes(out) != 0) {
        fprintf(stderr, "Expected the identity-equivalent gate to be removed.\n");
        result = EqualityError;
    }
    qk_dag_free(out);

cleanup_pm:
    qk_passmanager_free(pm);
    qk_ir_handle_free(dag_ir);
    // The pass manager took ownership of the pass; nothing else to free for it.
    return result;
}

int test_pass_library(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_builtin_pass_invalid_id);
    num_failed += RUN_TEST(test_builtin_remove_identity_equivalent);
    num_failed += RUN_TEST(test_builtin_remove_diagonal_gates_before_measure);
    num_failed += RUN_TEST(test_builtin_optimize_1q_sequences);
    num_failed += RUN_TEST(test_builtin_setters_reject_invalid_usage);
    num_failed += RUN_TEST(test_builtin_set_target_accepts_target);
    fflush(stderr);
    return num_failed;
}
