// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! A library of Qiskit's transpiler passes, exposed through the C API's pass-manager interface.
//!
//! Running one of Qiskit's passes from C previously required manually writing a config struct, an
//! execution function, and a vtable for every pass that a user wanted to run.  This module moves
//! that boilerplate into Qiskit itself: the config is [`BuiltinTranspilerPass`], the execution
//! function and vtable are predefined, and the whole thing is exposed to C through
//! [`qk_pass_builtin_new`] and the `qk_pass_builtin_set_*` configuration functions.

use std::ffi::{CStr, c_void};
use std::ptr;
use std::sync::LazyLock;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_passmanager::{PassContext, StaticPass};
use qiskit_transpiler::passes::{
    Optimize1qGatesDecompositionState, run_optimize_1q_gates_decomposition,
    run_remove_diagonal_before_measure, run_remove_identity_equiv,
};
use qiskit_transpiler::target::Target;

use crate::ExitCode;
use crate::dyn_types::VtableEntry;
use crate::passmanager::{
    CPass, CompilationError, IrBuiltin, IrHandle, PassSlot, qk_ir_handle_builtin, qk_pass_new,
    qk_pass_vtable_free, qk_pass_vtable_new,
};
use crate::pointers::ExposesOwnedPointers;

/// @ingroup pass-manager
/// Enumeration of the built-in transpiler passes available through `qk_pass_builtin_new`.
///
/// All of these passes take the `QkDag` built-in IR type on both input and output, so they can be
/// freely combined with each other in a pass-manager pipeline.
#[derive(Clone, Copy, derive_more::TryFrom, Debug, PartialEq, Eq)]
#[try_from(repr)]
#[repr(u32)]
pub enum PassBuiltin {
    /// The RemoveIdentityEquivalent pass.
    ///
    /// Removes gates whose effect is equivalent to the identity operation, up to a tolerance
    /// derived from the pass's target and approximation degree.  Configurable with
    /// `qk_pass_builtin_set_target` and `qk_pass_builtin_set_approximation_degree`.
    RemoveIdentityEquivalent = 0,
    /// The RemoveDiagonalGatesBeforeMeasure pass.
    ///
    /// Removes diagonal gates (including mid-circuit resets) that act on qubits which are only
    /// measured afterwards.  This pass has no configuration.
    RemoveDiagonalGatesBeforeMeasure = 1,
    /// The Optimize1qGatesDecomposition pass.
    ///
    /// Optimizes runs of single-qubit gates into shorter sequences over a basis.  Configurable
    /// with `qk_pass_builtin_set_target`.
    Optimize1qSequences = 2,
}

/// The configuration of a built-in transpiler pass.
///
/// This is the Rust-space "predefined struct" backing the C-API pass library: one variant per
/// pass, holding the same option data that the Python-space pass classes take as constructor
/// arguments.  Instances are created by `qk_pass_builtin_new` and are exposed to C as the `this`
/// data pointer of the pass's predefined vtable.
///
/// This implements [`StaticPass`] so it can also be used directly from first-party Rust code, for
/// example in pass managers built programmatically.
#[derive(Clone, Debug)]
pub enum BuiltinTranspilerPass {
    /// The RemoveIdentityEquivalent pass; see [`PassBuiltin::RemoveIdentityEquivalent`].
    RemoveIdentityEquivalent {
        /// The target to report gate error rates from.  If unset, the cutoff tolerance falls back
        /// to the minimum numerical tolerance.
        target: Option<Target>,
        /// The approximation degree, or `None` to use the target's error rates directly.
        approximation_degree: Option<f64>,
    },
    /// The RemoveDiagonalGatesBeforeMeasure pass; see
    /// [`PassBuiltin::RemoveDiagonalGatesBeforeMeasure`].
    RemoveDiagonalGatesBeforeMeasure,
    /// The Optimize1qGatesDecomposition pass; see [`PassBuiltin::Optimize1qSequences`].
    Optimize1qSequences {
        /// The target defining the basis gates and gate errors for the decomposition.  If unset,
        /// the pass optimizes over all known one-qubit basis sets.
        target: Option<Target>,
    },
}

impl BuiltinTranspilerPass {
    /// The canonical human-readable name of the pass.
    ///
    /// This matches the name of the corresponding Python-space transpiler pass, and is used for
    /// debugging output.
    fn name(&self) -> &'static CStr {
        match self {
            Self::RemoveIdentityEquivalent { .. } => c"RemoveIdentityEquivalent",
            Self::RemoveDiagonalGatesBeforeMeasure => c"RemoveDiagonalGatesBeforeMeasure",
            Self::Optimize1qSequences { .. } => c"Optimize1qGatesDecomposition",
        }
    }
}

impl StaticPass<DAGCircuit> for BuiltinTranspilerPass {
    fn run(
        &self,
        mut ir: Box<DAGCircuit>,
        _context: &mut PassContext,
    ) -> anyhow::Result<Box<DAGCircuit>> {
        match self {
            Self::RemoveIdentityEquivalent {
                target,
                approximation_degree,
            } => {
                run_remove_identity_equiv(&mut ir, *approximation_degree, target.as_ref())?;
            }
            Self::RemoveDiagonalGatesBeforeMeasure => {
                run_remove_diagonal_before_measure(&mut ir);
            }
            Self::Optimize1qSequences { target } => {
                let state = Optimize1qGatesDecompositionState::new(
                    target
                        .as_ref()
                        .and_then(|target| target.num_qubits)
                        .map(|n| n as usize)
                        .unwrap_or(0),
                );
                run_optimize_1q_gates_decomposition(&mut ir, &state, target.as_ref(), None, None)?;
            }
        }
        Ok(ir)
    }
}

/// The signature of the `QkPassSlot_RunOwned` vtable slot.
type RunOwnedFn = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut PassContext,
    *mut *mut CompilationError,
) -> *mut c_void;

/// The signature of the `QkPassSlot_Delete` vtable slot.
type DeleteFn = unsafe extern "C" fn(*mut c_void);

/// The `RunOwned` vtable slot shared by all built-in transpiler passes.
///
/// # Safety
///
/// `this` must be an owned pointer to a `BuiltinTranspilerPass` created by
/// [`qk_pass_builtin_new`], `ir` must be an owned `QkDag` pointer, and `context` must be a valid
/// `QkPassContext`.  The input IR is freed on all paths; on error, a `QkCompilationError` is
/// written to `error` (if non-`NULL`).
unsafe extern "C" fn run_builtin_transpiler_pass(
    this: *mut c_void,
    ir: *mut c_void,
    context: *mut PassContext,
    error: *mut *mut CompilationError,
) -> *mut c_void {
    // SAFETY: per the function documentation, `this` is a valid config pointer created by us.
    let config = unsafe { &*this.cast::<BuiltinTranspilerPass>() };
    // SAFETY: per the function documentation, `ir` is an owned `QkDag`.
    let ir = unsafe { DAGCircuit::steal(ir.cast()) };
    // SAFETY: per the function documentation, `context` is a valid context handle.
    let context = unsafe { &mut *context };
    match config.run(ir, context) {
        // `StaticPass::run` took ownership of the input IR, so it is freed in all paths,
        // including errors.
        Ok(ir) => ir.into_leaked().cast(),
        Err(err) => {
            if !error.is_null() {
                // SAFETY: per the function documentation, `error` is valid for a single write.
                unsafe { error.write(CompilationError(err).into_leaked()) };
            }
            ptr::null_mut()
        }
    }
}

/// The `Delete` vtable slot shared by all built-in transpiler passes.
///
/// # Safety
///
/// `this` must be `NULL`, or an owned `BuiltinTranspilerPass` pointer created by
/// [`qk_pass_builtin_new`] that has not been deleted yet.
unsafe extern "C" fn delete_builtin_transpiler_pass(this: *mut c_void) {
    // The no-op on `NULL` matches the documented `QkPassSlot_Delete` behaviour.
    if !this.is_null() {
        // SAFETY: per the function documentation, `this` is an owned config pointer.
        drop(unsafe { Box::from_raw(this.cast::<BuiltinTranspilerPass>()) });
    }
}

/// A wrapper around a `VtableEntry` array making it usable as a static.
///
/// SAFETY: the array contains only immutable function-pointer data that is safe to share between
/// threads; see the wrapped entries for their individual contracts.
#[derive(Clone, Copy)]
struct SyncVtableEntryTable([VtableEntry; 3]);
// SAFETY: the wrapped array contains only plain data and function pointers to `unsafe extern "C"`
// shims that are safe to call from any thread.
unsafe impl Sync for SyncVtableEntryTable {}

/// The vtable slot table for all built-in transpiler passes.
static BUILTIN_TRANSPILER_PASS_SLOTS: SyncVtableEntryTable = SyncVtableEntryTable([
    VtableEntry {
        slot: PassSlot::RunOwned as u32,
        flags: 0,
        ptr: (run_builtin_transpiler_pass as RunOwnedFn) as *mut c_void,
    },
    VtableEntry {
        slot: PassSlot::Delete as u32,
        flags: 0,
        ptr: (delete_builtin_transpiler_pass as DeleteFn) as *mut c_void,
    },
    VtableEntry {
        slot: u32::MAX,
        flags: 0,
        ptr: ptr::null_mut(),
    },
]);

// All built-in transpiler passes operate on the DAG IR, for both input and output.  This handle is
// cached once and borrowed for every vtable we construct.
static DAG_IR_HANDLE: LazyLock<IrHandle> = LazyLock::new(|| {
    // SAFETY: `qk_ir_handle_builtin` returns an owned handle pointer, which is valid to `steal`.
    *unsafe { IrHandle::steal(qk_ir_handle_builtin(IrBuiltin::Dag as u32)) }
});

/// Get the mutable config of a built-in pass, if `pass` is one.
///
/// Returns `NULL` if `pass` is not a built-in transpiler pass, so that the setters can reject
/// foreign passes rather than cast an unrelated data pointer.
///
/// # Safety
///
/// `pass` must be `NULL`, or point to a valid `QkPass`.
unsafe fn builtin_config(pass: *const CPass) -> Option<&'static mut BuiltinTranspilerPass> {
    if pass.is_null() {
        return None;
    }
    // SAFETY: per the function documentation, `pass` points to a valid `QkPass`.
    let pass = unsafe { &*pass };
    if !pass.is_run_by(run_builtin_transpiler_pass) {
        return None;
    }
    // SAFETY: the `RunOwned` shim of this pass is `run_builtin_transpiler_pass`, whose contract
    // is that `this` is a `BuiltinTranspilerPass` pointer.
    Some(unsafe { &mut *pass.data_ptr().cast::<BuiltinTranspilerPass>() })
}

/// @ingroup pass-manager
/// Create a new instance of one of Qiskit's built-in transpiler passes.
///
/// This is the entry point into the built-in pass library; the available passes are enumerated by
/// `QkPassBuiltin`.  The pass is created with its default configuration; some passes accept
/// further configuration through the `qk_pass_builtin_set_*` functions.
///
/// All built-in passes take the `QkDag` IR type on input and output, so they can be freely mixed
/// with each other in a pass-manager pipeline.
///
/// The returned object is a complete `QkPass`: it can be pushed onto a `QkPassManager` with
/// `qk_passmanager_push_pass`, or released directly with `qk_pass_free`.
///
/// @param ty An identifier for the desired pass.  See `QkPassBuiltin` for the allowed values.
/// @return An owned pass object, or `NULL` if given an invalid value.
///
/// # Example
///
/// ```c
/// QkPassManager *pm = qk_passmanager_new();
/// QkPass *pass = qk_pass_builtin_new(QkPassBuiltin_RemoveDiagonalGatesBeforeMeasure);
/// qk_passmanager_push_pass(pm, pass);
///
/// QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);
/// QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, dag, dag_ir, dag_ir, NULL);
///
/// qk_dag_free(out);
/// qk_passmanager_free(pm);
/// qk_ir_handle_free(dag_ir);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn qk_pass_builtin_new(ty: u32) -> *mut CPass {
    let config = match PassBuiltin::try_from(ty) {
        Ok(PassBuiltin::RemoveIdentityEquivalent) => {
            BuiltinTranspilerPass::RemoveIdentityEquivalent {
                target: None,
                approximation_degree: Some(1.0),
            }
        }
        Ok(PassBuiltin::RemoveDiagonalGatesBeforeMeasure) => {
            BuiltinTranspilerPass::RemoveDiagonalGatesBeforeMeasure
        }
        Ok(PassBuiltin::Optimize1qSequences) => {
            BuiltinTranspilerPass::Optimize1qSequences { target: None }
        }
        Err(_) => return ptr::null_mut(),
    };
    let ir = &raw const *DAG_IR_HANDLE;
    // SAFETY: `config.name()` is nul-terminated, `ir` points to a valid `IrHandle` (the
    // cached `QkIrBuiltin_Dag` handle), and the slot table is well-formed with no duplicates.
    let vtable = unsafe {
        qk_pass_vtable_new(
            config.name().as_ptr(),
            ir,
            ir,
            BUILTIN_TRANSPILER_PASS_SLOTS.0.as_ptr(),
        )
    };
    // SAFETY: the vtable's shims expect the `this` pointer to be a `BuiltinTranspilerPass`,
    // which is exactly what we pass.
    let pass = unsafe { qk_pass_new(Box::into_raw(Box::new(config)).cast(), vtable) };
    // SAFETY: `vtable` is the valid result of `qk_pass_vtable_new`; each pass owns its own
    // reference to it, so ours can be released immediately.
    unsafe { qk_pass_vtable_free(vtable) };
    pass
}

/// @ingroup pass-manager
/// Set the target of a built-in pass.
///
/// This configures the pass's target, in the same way as the `target` argument of the
/// corresponding Python-space pass.  The pass takes its own copy of the target, so the caller
/// remains the owner of the given object and may release it immediately afterwards.
///
/// This applies to the passes that document a target option; see `QkPassBuiltin`.  Passing a
/// `NULL` target clears the option.
///
/// @param pass A borrowed pointer to the pass to configure.
/// @param target A borrowed pointer to a `QkTarget`, or `NULL` to clear the target.
/// @return `QkExitCode_Success` on success, or `QkExitCode_InvalidPassOption` if `pass` is not a
///     built-in pass from `qk_pass_builtin_new`, or is a built-in pass that does not accept a
///     target.
///
/// # Safety
///
/// Behavior is undefined if `pass` is not either `NULL` or a valid pointer to a `QkPass`, or if
/// `target` is not either `NULL` or a valid pointer to a `QkTarget`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_builtin_set_target(
    pass: *const CPass,
    target: *const Target,
) -> ExitCode {
    // SAFETY: per documentation, `pass` is null or a valid `QkPass`.
    let Some(config) = (unsafe { builtin_config(pass) }) else {
        return ExitCode::InvalidPassOption;
    };
    // SAFETY: per documentation, `target` is null or a valid `QkTarget`.
    let target = unsafe { (!target.is_null()).then(|| (*target).clone()) };
    match config {
        BuiltinTranspilerPass::RemoveIdentityEquivalent { target: slot, .. }
        | BuiltinTranspilerPass::Optimize1qSequences { target: slot } => {
            *slot = target;
            ExitCode::Success
        }
        BuiltinTranspilerPass::RemoveDiagonalGatesBeforeMeasure => ExitCode::InvalidPassOption,
    }
}

/// @ingroup pass-manager
/// Set the approximation degree of a built-in pass.
///
/// This configures the pass's approximation degree, in the same way as the `approximation_degree`
/// argument of the corresponding Python-space pass.  The value can be a floating-point number
/// between 0 and 1 inclusive, or `NAN` to approximate up to the fidelity reported for the gate in
/// the pass's target.
///
/// This applies to the passes that document an approximation degree; see `QkPassBuiltin`.
///
/// @param pass A borrowed pointer to the pass to configure.
/// @param approximation_degree The approximation degree to set.
/// @return `QkExitCode_Success` on success, or `QkExitCode_InvalidPassOption` if `pass` is not a
///     built-in pass from `qk_pass_builtin_new`, is a built-in pass that does not accept an
///     approximation degree, or the given value is invalid.
///
/// # Safety
///
/// Behavior is undefined if `pass` is not either `NULL` or a valid pointer to a `QkPass`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_builtin_set_approximation_degree(
    pass: *const CPass,
    approximation_degree: f64,
) -> ExitCode {
    // SAFETY: per documentation, `pass` is null or a valid `QkPass`.
    let Some(config) = (unsafe { builtin_config(pass) }) else {
        return ExitCode::InvalidPassOption;
    };
    match config {
        BuiltinTranspilerPass::RemoveIdentityEquivalent {
            approximation_degree: slot,
            ..
        } => {
            *slot = if approximation_degree.is_nan() {
                None
            } else if (0.0..=1.0).contains(&approximation_degree) {
                Some(approximation_degree)
            } else {
                return ExitCode::InvalidPassOption;
            };
            ExitCode::Success
        }
        _ => ExitCode::InvalidPassOption,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qiskit_circuit::Clbit;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::StandardInstruction;
    use qiskit_circuit::operations::{Operation, Param, StandardGate};
    use qiskit_circuit::packed_instruction::PackedOperation;
    use qiskit_passmanager::PassManager;

    /// A `CircuitData` of one qubit holding the given gate sequence, converted to a DAG.
    ///
    /// Param-less gates must be pushed without a parameter; giving one would leave the DAG unable
    /// to produce the gate's matrix, which several transpiler passes rely on.
    fn dag_from_gates(gates: &[(StandardGate, f64)]) -> DAGCircuit {
        let mut qc = CircuitData::with_capacity(1, 0, gates.len(), Param::Float(0.0)).unwrap();
        for (gate, theta) in gates {
            let params: &[Param] = if gate.num_params() == 0 {
                &[]
            } else {
                &[Param::Float(*theta)]
            };
            qc.push_standard_gate(*gate, params, &[Qubit(0)]).unwrap();
        }
        DAGCircuit::from_circuit_data(&qc, false, None, None).unwrap()
    }

    #[test]
    fn pass_builtin_try_from() {
        assert_eq!(
            PassBuiltin::try_from(0).unwrap(),
            PassBuiltin::RemoveIdentityEquivalent
        );
        assert_eq!(
            PassBuiltin::try_from(1).unwrap(),
            PassBuiltin::RemoveDiagonalGatesBeforeMeasure
        );
        assert_eq!(
            PassBuiltin::try_from(2).unwrap(),
            PassBuiltin::Optimize1qSequences
        );
        assert!(PassBuiltin::try_from(3).is_err());
    }

    #[test]
    fn remove_identity_equivalent_removes_identity_equiv_gates() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(BuiltinTranspilerPass::RemoveIdentityEquivalent {
            target: None,
            approximation_degree: None,
        })
        .unwrap();
        // `rz(pi)` is diagonal, but `rx(1e-20)` is equivalent to the identity up to the
        // minimum numerical tolerance; only the latter should be removed.
        let dag = dag_from_gates(&[
            (StandardGate::H, 0.0),
            (StandardGate::RX, 1e-20),
            (StandardGate::RZ, std::f64::consts::PI),
            (StandardGate::H, 0.0),
        ]);
        let out: DAGCircuit = pm.run::<DAGCircuit, DAGCircuit>(dag).unwrap().0;
        let names: Vec<&str> = out.op_nodes(false).map(|(_, n)| n.op.name()).collect();
        assert_eq!(names, ["h", "rz", "h"]);
    }

    #[test]
    fn remove_diagonal_gates_before_measure_works() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(BuiltinTranspilerPass::RemoveDiagonalGatesBeforeMeasure)
            .unwrap();
        // A diagonal gate directly before a measure on the same qubit is removable; the
        // pre-measure `x` is not diagonal and must survive.
        let mut qc = CircuitData::with_capacity(1, 1, 3, Param::Float(0.0)).unwrap();
        qc.push_standard_gate(StandardGate::X, &[], &[Qubit(0)])
            .unwrap();
        qc.push_standard_gate(StandardGate::RZ, &[Param::Float(1.23)], &[Qubit(0)])
            .unwrap();
        qc.push_packed_operation(
            PackedOperation::from_standard_instruction(StandardInstruction::Measure),
            None,
            &[Qubit(0)],
            &[Clbit(0)],
        )
        .unwrap();
        let dag = DAGCircuit::from_circuit_data(&qc, false, None, None).unwrap();
        let out: DAGCircuit = pm.run::<DAGCircuit, DAGCircuit>(dag).unwrap().0;
        let names: Vec<&str> = out.op_nodes(false).map(|(_, n)| n.op.name()).collect();
        assert_eq!(names, ["x", "measure"]);
    }

    #[test]
    fn optimize_1q_sequences_reduces_gate_count() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(BuiltinTranspilerPass::Optimize1qSequences { target: None })
            .unwrap();
        // Three Hadamards on one qubit equal a single Hadamard up to global phase; the
        // optimizer should collapse the run to at most one gate.
        let dag = dag_from_gates(&[
            (StandardGate::H, 0.0),
            (StandardGate::H, 0.0),
            (StandardGate::H, 0.0),
        ]);
        let out: DAGCircuit = pm.run::<DAGCircuit, DAGCircuit>(dag).unwrap().0;
        assert_eq!(out.count_ops(false).unwrap().len(), 1);
    }

    #[test]
    fn pass_builtin_new_creates_and_frees_each_builtin() {
        for ty in 0..3 {
            let pass = qk_pass_builtin_new(ty);
            assert!(!pass.is_null());
            // SAFETY: the pointer is the owned result of `qk_pass_builtin_new`.
            unsafe { crate::passmanager::qk_pass_free(pass) };
        }
        assert!(qk_pass_builtin_new(3).is_null());
        assert!(qk_pass_builtin_new(u32::MAX).is_null());
    }

    #[test]
    fn setters_reject_invalid_usage() {
        // SAFETY: `pass` is NULL or a valid `QkPass` per the setter contracts.
        unsafe {
            // Invalid builtin id: NULL pass.
            assert_eq!(
                qk_pass_builtin_set_target(std::ptr::null(), std::ptr::null()),
                ExitCode::InvalidPassOption
            );
            // The diagonal-gates pass has no target option.
            let pass = qk_pass_builtin_new(PassBuiltin::RemoveDiagonalGatesBeforeMeasure as u32);
            assert_eq!(
                qk_pass_builtin_set_target(pass, std::ptr::null()),
                ExitCode::InvalidPassOption
            );
            assert_eq!(
                qk_pass_builtin_set_approximation_degree(pass, 0.5),
                ExitCode::InvalidPassOption
            );
            crate::passmanager::qk_pass_free(pass);

            // The identity-removal pass accepts approximation degrees, but only in [0, 1].
            let pass = qk_pass_builtin_new(PassBuiltin::RemoveIdentityEquivalent as u32);
            assert_eq!(
                qk_pass_builtin_set_approximation_degree(pass, 0.5),
                ExitCode::Success
            );
            assert_eq!(
                qk_pass_builtin_set_approximation_degree(pass, f64::NAN),
                ExitCode::Success
            );
            assert_eq!(
                qk_pass_builtin_set_approximation_degree(pass, 1.5),
                ExitCode::InvalidPassOption
            );
            // A NULL target clears the option.
            assert_eq!(
                qk_pass_builtin_set_target(pass, std::ptr::null()),
                ExitCode::Success
            );
            crate::passmanager::qk_pass_free(pass);
        }
    }
}
