# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
ParameterExpression Class to enable creating simple expressions of Parameters.
"""

from __future__ import annotations

from collections.abc import Mapping

from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.optionals import HAS_SYMPY
import qiskit._accelerate.circuit

Parameter = qiskit._accelerate.circuit.Parameter
ParameterExpression = qiskit._accelerate.circuit.ParameterExpression
OpCode = qiskit._accelerate.circuit.OpCode


_OP_CODE_MAP = (
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__pow__",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "exp",
    "log",
    "sign",
    "gradient",
    "conjugate",
    "subs",
    "abs",
    "arctan",
    "__rsub__",
    "__rtruediv__",
    "__rpow__",
)


def op_code_to_method(op_code: OpCode | int) -> str:
    """Return the method name for a given op_code."""
    return _OP_CODE_MAP[int(op_code)]


@HAS_SYMPY.require_in_call
def sympify(expression):
    """Return symbolic expression as a raw Sympy object.

    .. note::

        This is for interoperability only.  Qiskit will not accept or work with raw Sympy or
        Symengine expressions in its parameters, because they do not contain the tracking
        information used in circuit-parameter binding and assignment.

    .. warning::

        This method will pass strings from the expression to sympy which is
        inherently unsafe as sympy heavily relies on Python's :func:`.eval`
        function for the processing of strings. This method is not safe to
        use with untrusted input.
    """
    import sympy

    # if the expression is a plain Parameter, we just need to return the Symbol
    if isinstance(expression, Parameter):
        return sympy.Symbol(expression.name)
    elif expression.is_symbol():
        return sympy.Symbol(expression.parameters.pop().name)

    try:
        value = expression.numeric(strict=False)
        return sympy.Number(value)
    except TypeError as _:
        pass

    # Otherwise we rebuild the expression from the QPY replay. We keep track of expressions
    # in a stack, which will be queried if an lhs or rhs element is None (meaning that it must
    # be applied to the previous expression).
    stack = []
    for inst in expression._qpy_replay:
        for operand in [inst.lhs, inst.rhs]:
            if operand is not None:
                if isinstance(operand, ParameterExpression):
                    stack.append(operand.sympify())
                else:
                    stack.append(operand)

        method_str = op_code_to_method(inst.op)
        # checks if we apply a binary operation, requiring lhs and rhs
        if inst.op in {
            OpCode.ADD,
            OpCode.SUB,
            OpCode.MUL,
            OpCode.DIV,
            OpCode.POW,
            OpCode.RSUB,
            OpCode.RDIV,
            OpCode.RPOW,
        }:
            rhs = stack.pop()
            lhs = stack.pop()

            # Handle reverse operations (RSUB, RDIV, RPOW) by swapping operands
            if inst.op in {OpCode.RSUB, OpCode.RDIV, OpCode.RPOW}:
                # For reverse operations, the Rust code already swapped operands in the replay,
                # so we need to apply the operation with rhs op lhs to get the correct result.
                # For RPOW: rhs ** lhs (Rust already swapped, so we use __pow__ directly)
                # For RDIV: rhs / lhs (construct division directly: 2 / a)
                # For RSUB: rhs - lhs (construct subtraction directly: 2 - a)
                if inst.op == OpCode.RPOW:
                    # Rust already swapped operands, so we use __pow__ directly: rhs ** lhs
                    stack.append(rhs**lhs)
                elif inst.op == OpCode.RDIV:
                    # Construct division directly: rhs / lhs (e.g., 2 / a)
                    # Use sympy's division operator to handle the case where
                    # __rtruediv__ returns NotImplemented
                    stack.append(rhs / lhs)
                elif inst.op == OpCode.RSUB:
                    # Construct subtraction directly: rhs - lhs (e.g., 2 - a)
                    # Use sympy's subtraction operator to handle the case where
                    # __rsub__ returns NotImplemented
                    stack.append(rhs - lhs)
            elif (
                not isinstance(lhs, sympy.Basic)
                and isinstance(rhs, sympy.Basic)
                and inst.op in [OpCode.ADD, OpCode.MUL]
            ):
                if inst.op == OpCode.ADD:
                    method_str = "__radd__"
                elif inst.op == OpCode.MUL:
                    method_str = "__rmul__"
                stack.append(getattr(rhs, method_str)(lhs))
            else:
                stack.append(getattr(lhs, method_str)(rhs))

        # these are unary operands, which only require a lhs operand
        else:
            lhs = stack.pop()

            if inst.op == OpCode.ACOS:
                stack.append(getattr(sympy, "acos")(lhs))
            elif inst.op == OpCode.ASIN:
                stack.append(getattr(sympy, "asin")(lhs))
            elif inst.op == OpCode.ATAN:
                stack.append(getattr(sympy, "atan")(lhs))
            elif inst.op == OpCode.ABS:
                stack.append(getattr(sympy, "Abs")(lhs))
            else:
                stack.append(getattr(sympy, method_str)(lhs))

    return stack.pop()


# Mapping from the names of Sympy functions that can appear in a Sympy expression to the
# corresponding method of :class:`.ParameterExpression`.
_SYMPY_FUNCTION_MAP = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "exp": "exp",
    "log": "log",
    "Abs": "abs",
    "sign": "sign",
    "conjugate": "conjugate",
}


@HAS_SYMPY.require_in_call
def from_sympy(
    expression,
    parameters: Mapping | None = None,
):
    """Convert a Sympy expression into a :class:`.ParameterExpression`.

    This is the inverse of :func:`.sympify`: it rebuilds a native Qiskit
    :class:`.ParameterExpression` out of a raw Sympy expression. The Sympy symbols in the
    expression are matched to :class:`.Parameter` instances by name.

    Args:
        expression: A Sympy expression. It must contain only :class:`~sympy.Symbol`\\ s, numbers
            and the operations supported by :class:`.ParameterExpression` (arithmetic, powers and
            ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``exp``, ``log``, ``Abs``,
            ``sign`` and ``conjugate``).
        parameters: An optional mapping of Sympy symbol names (or Sympy symbols) to the
            :class:`.Parameter` instances they should be represented by. This can be used to
            match symbols to existing :class:`.Parameter` objects, for example those of a
            :class:`.ParameterVector`. A mapping value may also be a
            :class:`.ParameterExpression` or a real or complex number, in which case the symbol
            is substituted with it. Any symbol not present in the mapping is replaced by a
            fresh :class:`.Parameter` with the same name.

    Returns:
        The converted :class:`.ParameterExpression`.

    Raises:
        CircuitError: If the expression is not a Sympy expression, contains Sympy functions
            that have no :class:`.ParameterExpression` equivalent, or the ``parameters``
            mapping contains values of an unsupported type.
    """
    import sympy

    if not isinstance(expression, sympy.Expr):
        raise CircuitError(
            "The expression to convert must be a Sympy expression, "
            f"not '{type(expression).__name__}'."
        )

    symbol_map: dict[str, Parameter | ParameterExpression | int | float | complex] = {}
    if parameters is not None:
        for key, value in parameters.items():
            if not isinstance(value, (Parameter, ParameterExpression, int, float, complex)):
                raise CircuitError(
                    f"Cannot map the Sympy symbol '{key}' to an object of type "
                    f"'{type(value).__name__}'; expected a Parameter, ParameterExpression or "
                    "a number."
                )
            name = key.name if isinstance(key, sympy.Symbol) else str(key)
            symbol_map[name] = value

    def _power(base, exp):
        """Build ``base ** exp`` for the raw Sympy exponent ``exp``, avoiding no-op powers of
        one and rebuilding negative powers as divisions; symbolic exponents are converted."""
        if exp.is_Number:
            exp = int(exp) if exp.is_Integer else float(exp)
            if exp < 0:
                out = base
                if exp < -1:
                    out = base ** (-exp)
                return 1 / out
            if exp == 1:
                return base
            return base**exp
        return base ** convert(exp)

    def convert(node):
        if isinstance(node, sympy.Symbol):
            name = node.name
            if (param := symbol_map.get(name)) is None:
                param = symbol_map[name] = Parameter(name)
            return param
        if isinstance(node, sympy.Number):
            # This covers ``Integer``, ``Float``, ``Rational`` and things like ``nan`` or ``oo``.
            if node.is_Integer:
                return int(node)
            if node.is_Float:
                return float(node)
            return float(node)
        if isinstance(node, sympy.NumberSymbol):
            # Irrational constants like ``pi`` and ``E`` are approximated numerically, as
            # :class:`.ParameterExpression` has no exact representation for them.
            return float(node)
        if node is sympy.I:
            return complex(node)
        if isinstance(node, sympy.Add):
            terms = [convert(term) for term in node.args]
            out = terms[0]
            for term in terms[1:]:
                out = out + term
            return out
        if isinstance(node, sympy.Mul):
            # A Sympy ``Mul`` represents division as a multiplication by a negative power (for
            # symbols) or by a non-integer rational coefficient (for numbers), so separate the
            # arguments into numerators and denominators to rebuild the divisions.
            numerator, denominator = [], []
            coeff_numerator, coeff_denominator = 1, 1
            for arg in node.args:
                if isinstance(arg, sympy.Rational):
                    if arg.is_Integer:
                        numerator.append(convert(arg))
                    else:
                        coeff_numerator *= int(arg.p)
                        coeff_denominator *= int(arg.q)
                elif isinstance(arg, sympy.Pow) and arg.exp.is_Number and arg.exp.is_negative:
                    denominator.append((arg.base, -arg.exp))
                else:
                    numerator.append(convert(arg))
            out = coeff_numerator
            for term in numerator:
                out = out * term
            if coeff_denominator != 1 or denominator:
                denominator_expr = coeff_denominator
                for base, exp in denominator:
                    denominator_expr = denominator_expr * _power(convert(base), exp)
                out = out / denominator_expr
            return out
        if isinstance(node, sympy.Pow):
            return _power(convert(node.base), node.exp)
        if isinstance(node, sympy.Expr) and node.func.__name__ in _SYMPY_FUNCTION_MAP:
            if len(node.args) != 1:
                raise CircuitError(
                    f"Cannot convert the Sympy function '{node.func.__name__}' with "
                    f"{len(node.args)} arguments to a ParameterExpression."
                )
            return getattr(convert(node.args[0]), _SYMPY_FUNCTION_MAP[node.func.__name__])()
        raise CircuitError(
            f"Cannot convert the Sympy expression '{node}' of type '{type(node).__name__}' "
            "to a ParameterExpression; the type is not supported."
        )

    result = convert(expression)
    if isinstance(result, (int, float, complex)):
        result = ParameterExpression._Value(result)
    return result


# Redefine the type so external imports get an evaluated reference; Sphinx needs this to understand
# the type hints.
ParameterValueType = ParameterExpression | float
