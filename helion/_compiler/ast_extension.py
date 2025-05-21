from __future__ import annotations

import ast
import enum
import threading
import typing
from typing import TYPE_CHECKING
from typing import TypeVar

from .. import exc
from .source_location import SourceLocation
from .source_location import current_location

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .type_propagation import TypeInfo

    _T = TypeVar("_T", bound=ast.AST)
    _R = TypeVar("_R")

    class _TLS(typing.Protocol):
        active_nodes: list[ExtendedAST]


# Thread-local storage for tracking active AST nodes during processing.
# `tls.active_nodes` holds a list (stack) of `ExtendedAST` nodes
# that are currently in context (e.g., being visited or constructed).
tls: _TLS = typing.cast("_TLS", threading.local())


class LoopType(enum.Enum):
    """
    Enumerates the types of loops that can be encountered in the AST,
    distinguishing between host-side execution and device-side (grid/kernel) execution.
    """

    UNSET = enum.auto()
    HOST = enum.auto()
    GRID = enum.auto()
    DEVICE = enum.auto()


class ExtendedAST:
    """
    A mixin class that enhances standard Python AST nodes with additional
    metadata and functionality crucial for the Helion compiler.

    This class is not used directly but is mixed into dynamically created
    subclasses of `ast.AST` nodes (e.g., `ast.Name`, `ast.Call`).
    This allows Helion to associate source locations, type information,
    loop context, and other compiler-specific data with each node in the
    abstract syntax tree.

    Attributes:
        _location (SourceLocation): The source code location (file, line, column)
            from which this AST node originated.
        _type_info (TypeInfo | None): Propagated type information for this node.
            `None` if type information has not been determined or is not applicable.
        _loop_type (LoopType): Indicates the type of loop this node is part of
            (e.g., HOST, GRID, DEVICE). Defaults to `LoopType.UNSET`.
        _is_kernel_call (bool): True if this AST node represents a call to another
            Helion kernel. Defaults to `False`.
        _fields (tuple[str, ...]): Inherited from `ast.AST`, lists the names of
            the children nodes.
    """

    # pyre-ignore[13]
    _fields: tuple[str, ...]

    def __init__(
        self,
        *,
        _location: SourceLocation,
        _type_info: TypeInfo | None = None,
        _loop_type: LoopType = LoopType.UNSET,
        _is_kernel_call: bool = False,
        **kwargs: object,
    ) -> None:
        """
        Initializes the ExtendedAST mixin attributes.

        Args:
            _location: The source location of this AST node.
            _type_info: Optional type information for this node.
            _loop_type: The type of loop this node is part of.
            _is_kernel_call: Whether this node represents a kernel call.
            **kwargs: Arguments passed to the underlying `ast.AST` constructor.
        """

        super().__init__(**kwargs)
        self._type_info: TypeInfo | None = _type_info
        self._location: SourceLocation = _location
        self._loop_type: LoopType = _loop_type
        self._is_kernel_call: bool = _is_kernel_call

    def new(self, fields: dict[str, object]) -> ExtendedAST:
        """
        Creates a new instance of this node's class with updated fields,
        preserving existing metadata like location and type info.

        Args:
            fields: A dictionary of AST field names to their new values.

        Returns:
            A new ExtendedAST node of the same type.
        """
        result = self.__class__(
            **fields,
            _location=self._location,
            _type_info=self._type_info,
            _loop_type=self._loop_type,
            _is_kernel_call=self._is_kernel_call,
        )
        return self._location.to_ast(result)

    def fields(self) -> dict[str, object]:
        """
        Returns a dictionary of the standard AST fields and their current values for this node.

        Returns:
            A dictionary mapping field names (from `_fields`) to their values.
        """
        return {field: getattr(self, field) for field in self._fields}

    def copy(self, **changes: object) -> ExtendedAST:
        """
        Creates a shallow copy of this node, allowing specific fields to be
        overridden. Metadata like location and type info are preserved from the original.

        Args:
            **changes: Keyword arguments where keys are field names and values
                       are the new values for those fields.

        Returns:
            A new ExtendedAST node, which is a modified copy of the original.
        """
        return self.new({**self.fields(), **changes})

    def __repr__(self) -> str:
        """
        Returns a string representation of the AST node using `ast.dump`.
        """
        assert isinstance(self, ast.AST)
        return ast.dump(self)

    def update_type_info(self, type_info: TypeInfo) -> TypeInfo:
        """
        Updates the type information for this node. If existing type information
        is present, the new information is merged with it.

        Args:
            type_info: The new `TypeInfo` to associate with this node.
        """
        if self._type_info is not None and type_info != self._type_info:
            type_info = self._type_info.merge(type_info)
        self._type_info = type_info
        return self._type_info

    def debug_annotations(self) -> list[str]:
        """
        Generates a list of strings representing debug annotations for this node,
        such as its type information and loop type.

        Returns:
            A list of debug annotation strings.
        """
        result = []
        if self._type_info:
            result.extend(self._type_info.debug_annotations())
        if self._loop_type != LoopType.UNSET:
            result.append(f"loop_type={self._loop_type.name}")
        return result

    def __enter__(self) -> None:
        """
        Enters a context for this node, adding it to the thread-local stack
        of active nodes and entering its source location context.
        """
        try:
            tls.active_nodes.append(self)
        except AttributeError:
            tls.active_nodes = [self]
        self._location.__enter__()

    def __exit__(self, *args: object) -> None:
        """
        Exits the context for this node, removing it from the thread-local
        stack of active nodes and exiting its source location context.
        """
        self._location.__exit__(*args)
        tls.active_nodes.pop()

    @staticmethod
    def current() -> Sequence[ExtendedAST]:
        """
        Returns the current stack of active `ExtendedAST` nodes being processed
        in the current thread.
        """
        try:
            return tls.active_nodes
        except AttributeError:
            tls.active_nodes = rv = []
            return rv


# Cache mapping original `ast.AST` types to their `ExtendedAST`-wrapped counterparts.
# This avoids redundant creation of wrapper classes.
_to_extended: dict[type[ast.AST], type[ast.AST]] = {}


def get_wrapper_cls(cls: type[ast.AST]) -> type[ast.AST]:
    """
    Retrieves or creates a wrapper class for a given `ast.AST` subclass
    that includes `ExtendedAST` as a mixin.

    Args:
        cls: The original `ast.AST` class (e.g., `ast.Name`, `ast.FunctionDef`).

    Returns:
        The corresponding wrapper class that inherits from both `ExtendedAST` and `cls`.
    """
    if new_cls := _to_extended.get(cls):
        return new_cls

    class Wrapper(ExtendedAST, cls):
        pass

    Wrapper.__name__ = cls.__name__
    rv = typing.cast("type[ast.AST]", Wrapper)
    _to_extended[cls] = rv
    return rv


def create(cls: type[_T], **fields: object) -> _T:
    """
    Factory function to create an instance of an `ExtendedAST`-wrapped AST node.

    The node's `_location` attribute is automatically set using `current_location()`.

    Args:
        cls: The `ast.AST` class to instantiate (e.g., `ast.Call`, `ast.Assign`).
        **fields: Keyword arguments representing the fields of the AST node.

    Returns:
        An instance of the `ExtendedAST`-wrapped version of `cls`.
    """
    # pyre-ignore[28]
    result = get_wrapper_cls(cls)(**fields, _location=current_location())
    assert isinstance(result, ExtendedAST)
    result._location.to_ast(result)
    return typing.cast("_T", result)


def create_arg(name: str, annotation: str | None = None) -> ast.arg:
    """
    Helper function to create an `ExtendedAST`-wrapped `ast.arg` node.

    Args:
        name: The name of the argument.
        annotation: An optional type annotation string for the argument.

    Returns:
        An `ast.arg` node.
    """
    return create(
        ast.arg,
        arg=name,
        annotation=expr_from_string(annotation) if annotation else None,
        type_comment=None,
    )


def create_arguments(args: list[ast.arg]) -> ast.arguments:
    """
    Helper function to create an `ExtendedAST`-wrapped `ast.arguments` node.

    Args:
        args: A list of `ast.arg` nodes for positional arguments.

    Returns:
        An `ast.arguments` node.
    """
    return create(
        ast.arguments,
        args=args,
        posonlyargs=[],
        defaults=[],
        kw_defaults=[],
        kwonlyargs=[],
    )


def statement_from_string(template: str, **placeholders: ast.AST) -> ast.stmt:
    """
    Parses a string template into a single `ExtendedAST`-wrapped AST statement.

    Placeholders in the template (e.g., names like `VAR_NAME`) can be
    substituted with provided `ast.AST` nodes. All nodes in the resulting
    statement are converted to `ExtendedAST` instances.

    Args:
        template: A string containing Python code for a single statement.
        **placeholders: Keyword arguments where keys are placeholder names
                        in the template and values are `ast.AST` nodes to substitute.
    Returns:
        An `ast.stmt` node (wrapped with `ExtendedAST`).
    """
    (statement,) = ast.parse(template).body
    location: SourceLocation = current_location()

    def _replace(node: _R) -> _R:
        if isinstance(node, list):
            # pyre-ignore[7]
            return [_replace(item) for item in node]
        if not isinstance(node, ast.AST):
            return node
        if isinstance(node, ast.Name) and node.id in placeholders:
            # pyre-ignore[7]
            return placeholders[node.id]
        cls = get_wrapper_cls(type(node))
        # pyre-ignore[28]
        return location.to_ast(
            cls(
                **{field: _replace(getattr(node, field)) for field in node._fields},
                _location=location,
            )
        )
    # pyre-ignore[7]
    return _replace(statement)


def expr_from_string(template: str, **placeholders: ast.AST) -> ast.AST:
    """
    Parses a string template into an `ExtendedAST`-wrapped AST expression.

    Similar to `statement_from_string`, but expects the template to represent
    a single expression.

    Args:
        template: A string containing Python code for a single expression.
        **placeholders: Keyword arguments for placeholder substitution.

    Returns:
        An `ast.AST` node representing the expression (wrapped with `ExtendedAST`).
    """
    expr = statement_from_string(template, **placeholders)
    assert isinstance(expr, ast.Expr)
    return expr.value


def convert(node: ast.AST) -> ast.AST:
    """
    Recursively converts a standard Python `ast.AST` node (or a list of them)
    into its `ExtendedAST`-wrapped equivalent.

    Source location information is extracted from the original nodes if available,
    otherwise the current location is used.
    """
    if isinstance(node, ast.AST):
        cls = get_wrapper_cls(type(node))
        if "lineno" in node._attributes:
            location = SourceLocation.from_ast(node)
        else:
            # some nodes like arguments lack location information
            location = current_location()
        with location:
            # pyre-ignore[28]
            return cls(
                **{field: convert(getattr(node, field)) for field in node._fields},
                **{attr: getattr(node, attr) for attr in node._attributes},
                _location=location,
            )
    elif isinstance(node, list):
        # pyre-ignore[7]
        return [convert(item) for item in node]
    else:
        return node


class NodeVisitor(ast.NodeVisitor):
    """
    A custom AST node visitor that operates on `ExtendedAST` instances.

    This visitor extends Python's `ast.NodeVisitor` to ensure that the
    context management features of `ExtendedAST` (specifically `__enter__`
    and `__exit__` for tracking active nodes) are correctly invoked during
    tree traversal. It also provides more specific error handling for
    exceptions raised during visitation.

    When `visit` is called, it prints debugging information about the node,
    including its class, location, type info, loop type, kernel call status,
    and a summary of its AST fields.
    """

    def visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        print(f"Visiting node: {node.__class__.__name__}")
        print(f"  Location: {getattr(node, '_location', 'N/A')}")
        print(f"  Type Info: {getattr(node, '_type_info', 'N/A')}")
        print(f"  Loop Type: {getattr(node, '_loop_type', 'N/A')}")
        print(f"  Is Kernel Call: {getattr(node, '_is_kernel_call', 'N/A')}")
        if hasattr(node, '_fields'):
            print("  AST Fields:")
            for field_name in node._fields:
                field_value = getattr(node, field_name, 'N/A')
                if isinstance(field_value, ExtendedAST):
                    value_repr = f"ExtendedAST.{field_value.__class__.__name__}"
                elif isinstance(field_value, ast.AST):
                    value_repr = f"ast.{field_value.__class__.__name__}"
                elif isinstance(field_value, list):
                    value_repr = f"[List ({len(field_value)}) of {', '.join(type(x).__name__ for x in field_value[:3])}{'...' if len(field_value) > 3 else ''}]"
                else:
                    value_repr = repr(field_value)
                print(f"    {field_name}: {value_repr}")
        print("-" * 20)  # Separator
        with node:
            try:
                visitor = getattr(
                    self,
                    f"visit_{node.__class__.__name__}",
                    self.generic_visit,
                )
                # pyre-ignore[29]
                return visitor(node)
            except exc.Base:
                raise
            except Exception as e:
                raise exc.InternalError(e) from e


# --- Start of tuple assignment unparsing compatibility ---
# Determine whether vanilla ast.unparse keeps parentheses in expressions like "(a, b) = c".
# This behavior changed between Python versions (e.g., Python 3.11 keeps them,
# Python 3.12+ removes them in assignment contexts).
# If parentheses are kept by the default unparser for tuple assignments,
# Helion's custom `_TupleParensRemovedUnparser` will remove them to ensure
# consistent code generation, particularly for Triton, which might be sensitive
# to such syntactic sugar.
_test_src: str = "(a, b) = c"
_needs_to_remove_tuple_parens: bool = (
    ast.unparse(ast.parse(_test_src)).lstrip().startswith("(")
)
# --- End of tuple assignment unparsing compatibility ---


class _TupleParensRemovedUnparser(ast._Unparser):  # pyre-ignore[11]
    """
    A custom AST unparser that modifies the behavior of `ast._Unparser.visit_Tuple`
    to ensure consistent source code formatting for tuple assignments across
    different Python versions.

    Specifically, it removes parentheses from tuple assignments like `(a, b) = c`
    when unparsing on Python versions (e.g., 3.10, 3.11) that would otherwise
    keep them, to match the behavior of Python 3.12+ which omits them.
    This helps in generating consistent Triton code.
    """
    def visit_Tuple(self, node) -> None:  # pyre-ignore[2]
        if _needs_to_remove_tuple_parens and isinstance(
            getattr(node, "ctx", None), ast.Store
        ):
            if len(node.elts) == 1:  # single-element tuple
                self.traverse(node.elts[0])  # pyre-ignore[16]
                self.write(",")  # pyre-ignore[16]
            else:  # multi-element tuple
                self.interleave(  # pyre-ignore[16]
                    lambda: self.write(", "), self.traverse, node.elts
                )
            return
        # For everything else fall back to default behavior
        super().visit_Tuple(node)  # pyre-ignore[16]


def unparse(ast_obj: ast.AST) -> str:
    """
    Converts an `ExtendedAST` object (or a standard `ast.AST`) back into a
    Python source code string.

    Uses `_TupleParensRemovedUnparser` to ensure consistent formatting of
    tuple assignments.

    Args:
        ast_obj: The AST node to unparse.
    """
    unparser = _TupleParensRemovedUnparser()
    return unparser.visit(ast_obj)  # pyre-ignore[16]
