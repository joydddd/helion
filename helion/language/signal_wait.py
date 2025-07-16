from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.indexing_strategy import SubscriptIndexing
from . import _decorators

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["signal", "wait"]


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def wait(
    signal_pad: torch.Tensor,
    index: list[object] | None = None,
    signal: int = 1,
    update: int | None = None,
    scope: str = "gpu",
    hasSubsequentMemAccess: bool = True,
    as_ptrs: bool = False,
) -> None:
    """Wait until all entries of the signal_pad slice are equal to the signal value.
    Args:
        signal_pad: The signal pad tensor to wait on
        index: Indices to index into the signal_pad tensor
        signal: the value to wait for
        update: Atomically update the signal_pad tensor with this value once the signal is observed. (default: None)
        scope: The scope of the lock (default: 'gpu')
        as_ptrs: Treat signal_pad as pointers to global memory barriers (default: False)
        hasSubsequentMemAccess: Whether the wait is followed by a subsequence memory access (default: True)

    Returns:
        None
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(wait)
def _(
    signal_pad: torch.Tensor,
    index: list[object] | None = None,
    signal: int = 1,
    update: int | None = None,
    scope: str = "gpu",
    hasSubsequentMemAccess: bool = True,
    as_ptrs: bool = False,
) -> tuple[torch.Tensor, object, int, int | None, str, str, bool, bool]:
    from .tile_proxy import Tile

    valid_scopes = {"sys", "gpu"}

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    if as_ptrs:
        if index is not None:
            raise ValueError(
                f"When as_ptrs=True, signal_pad must be used without indexing. "
                f"Expected 0 indices but got {len(index)}. "
            )
        if signal_pad.dtype not in (torch.uint64, torch.int64):
            raise ValueError(
                f"When as_ptrs=True, signal_pad must have dtype torch.uint64 or torch.int64 "
                f"to represent memory pointers. Got dtype {signal_pad.dtype}. "
            )
    if index is None:
        index = []

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    return (signal_pad, index, signal, update, scope, has_subsequent_load, as_ptrs)


@_decorators.register_fake(wait)
def _(
    signal_pad: torch.Tensor,
    index: list[object] | None = None,
    signal: int = 1,
    update: int | None = None,
    scope: str = "gpu",
    hasSubsequentMemAccess: bool = True,
    as_ptrs: bool = False,
) -> None:
    return None


@_decorators.codegen(wait)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.indexing_strategy import SubscriptIndexing

    signal_pad = state.proxy_arg(0)
    index = state.proxy_arg(1)
    signal = state.proxy_arg(2)
    update = state.proxy_arg(3)
    scope = state.proxy_arg(4)
    has_subsequent_load = state.proxy_arg(5)
    as_ptrs = state.proxy_arg(6)

    assert isinstance(signal_pad, torch.Tensor)
    assert isinstance(index, (list))

    assert type(scope) is str

    assert type(has_subsequent_load) is bool
    assert type(as_ptrs) is bool

    sem = "acquire" if has_subsequent_load else "relaxed"
    op = "atomic_cas" if update is not None else "atomic_xchg"
    skip_sync = not has_subsequent_load

    if as_ptrs:
        bar_tensor_shape = signal_pad.shape
        bar_addrs = "signal_pad_arg.to(tl.pointer_type(tl.int32))"
    else:
        indices = SubscriptIndexing.create(state, signal_pad, index)
        if signal_pad.dtype not in (torch.int32, torch.uint32):
            raise NotImplementedError(
                f"Unsupported signal pad dtype: {signal_pad.dtype}. Must be of torch.int32 or torch.uint32."
            )
        signal_pad_name = state.device_function.tensor_arg(signal_pad).name
        bar_tensor_shape = SubscriptIndexing.compute_shape(signal_pad, index)
        bar_addrs = f"{signal_pad_name} + signal_pad_arg"

    signal_expr = ast.Constant(value=signal)  # pyright: ignore[reportArgumentType]
    update_expr = ast.Constant(value=update)  # pyright: ignore[reportArgumentType]

    is_scalar = len(bar_tensor_shape) == 0

    call_triton_wait_signal = f"helion.runtime.triton_wait_{'' if is_scalar else 'multiple_'}signal(addr={bar_addrs}, expect=signal, update=update, sem='{sem}', scope='{scope}', op='{op}', skip_sync={skip_sync})"

    return expr_from_string(
        call_triton_wait_signal,
        signal_pad_arg=state.ast_arg(0) if as_ptrs else indices.index_expr,  # pyright: ignore[reportPossiblyUnboundVariable]
        signal=signal_expr,
        update=update_expr,
    )


@has_side_effect
@_decorators.api(tiles_as_sizes=True, allow_host_tensor=True)
def signal(
    signal_pad: torch.Tensor,
    index: list[object] | None = None,
    signal: int = 1,
    wait_for: int | None = None,
    op: str | None = None,
    scope: str = "gpu",
    hasPreviousMemAccess: bool = True,
    as_ptrs: bool = False,
) -> torch.Tensor:
    """Set the signal_pad slice to the signal value.
    Args:
        signal_pad: The signal pad to signal
        index: Indices to index into the signal_pad tensor
        signal: the value to send
        wait_for: The value to wait for before sending the signal.
        op: The operating for updating the lock: "add", "set" (default: "set")
        scope: The scope of the lock (default: 'gpu')
        hasPreviousMemAccess: Whether the signal is preceded by a memory access (default: True)
        as_ptrs: Treat signal_pad as pointers to global memory barriers (default: False)
    Returns:
        The old value of the signal_pad slice before the update.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(signal)
def _(
    signal_pad: torch.Tensor,
    index: list[object] | None = None,
    signal: int = 1,
    wait_for: int | None = None,
    op: str | None = None,
    scope: str = "gpu",
    hasPreviousMemAccess: bool = True,
    as_ptrs: bool = False,
) -> tuple[torch.Tensor, object, int, int | None, str, str, bool, bool]:
    from .tile_proxy import Tile

    valid_ops = {"add", "set"}
    valid_scopes = {"sys", "gpu"}

    if op is None:
        op = "set"

    if op not in valid_ops:
        raise ValueError(f"Invalid signal op '{op}'. Must be one of {valid_ops}. ")

    if sem not in valid_sems:
        raise ValueError(
            f"Invalid memory semantic '{sem}'. Must be one of {valid_sems}."
        )

    if scope not in valid_scopes:
        raise ValueError(f"Invalid scope '{scope}'. Must be one of {valid_scopes}.")

    if as_ptrs:
        if index is not None:
            raise ValueError(
                f"When as_ptrs=True, signal_pad must be used without indexing. "
                f"Expected 0 indices but got {len(index)}. "
            )
        if signal_pad.dtype not in (torch.uint64, torch.int64):
            raise ValueError(
                f"When as_ptrs=True, signal_pad must have dtype torch.uint64 or torch.int64 "
                f"to represent memory pointers. Got dtype {signal_pad.dtype}. "
            )
    if index is None:
        index = []

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes(index)

    return (
        signal_pad,
        index,
        signal,
        wait_for,
        op,
        scope,
        hasPreviousMemAccess,
        as_ptrs,
    )


@_decorators.register_fake(signal)
def _(
    signal_pad: torch.Tensor,
    index: list[object] | None = None,
    signal: int = 1,
    wait_for: int | None = None,
    op: str | None = None,
    scope: str = "gpu",
    hasPreviousMemAccess: bool = True,
    as_ptrs: bool = False,
) -> torch.Tensor:
    if index is None:
        index = []
    if as_ptrs:
        return signal_pad.new_empty(signal_pad.shape)
    return signal_pad.new_empty(SubscriptIndexing.compute_shape(signal_pad, index))


@_decorators.codegen(signal)
def _(state: CodegenState) -> ast.AST:
    import ast

    from .._compiler.ast_extension import expr_from_string
    from .._compiler.indexing_strategy import SubscriptIndexing

    signal_pad = state.proxy_arg(0)
    index = state.proxy_arg(1)
    signal = state.proxy_arg(2)
    wait_for = state.proxy_arg(3)
    op = state.proxy_arg(4)
    scope = state.proxy_arg(5)
    hasPreviousMemAccess = state.proxy_arg(6)
    as_ptrs = state.proxy_arg(7)

    assert isinstance(signal_pad, torch.Tensor)
    assert isinstance(index, list)

    assert type(op) is str
    assert type(scope) is str

    assert type(hasPreviousMemAccess) is bool
    assert type(as_ptrs) is bool

    sem = "release" if hasPreviousMemAccess else "relaxed"
    skip_sync = not hasPreviousMemAccess

    if op == "add":
        op = "atomic_add"
    elif op == "set":
        op = "atomic_xchg" if wait_for is None else "atomic_cas"
    else:
        raise NotImplementedError(
            f"Unsupported op '{op}' for send signal on gmem barrier. "
        )

    if as_ptrs:
        bar_tensor_shape = signal_pad.shape
        bar_addrs = "signal_pad_arg.to(tl.pointer_type(tl.int32))"
    else:
        indices = SubscriptIndexing.create(state, signal_pad, index)
        if signal_pad.dtype not in (torch.int32, torch.uint32):
            raise NotImplementedError(
                f"Unsupported signal pad dtype: {signal_pad.dtype}. Must be of torch.int32 or torch.uint32."
            )
        signal_pad_name = state.device_function.tensor_arg(signal_pad).name
        bar_tensor_shape = SubscriptIndexing.compute_shape(signal_pad, index)
        bar_addrs = f"{signal_pad_name} + signal_pad_arg"

    is_scalar = len(bar_tensor_shape) == 0

    signal_expr = ast.Constant(value=signal)  # pyright: ignore[reportArgumentType]
    if wait_for is not None:
        wait_for_expr = ast.Constant(value=wait_for)  # pyright: ignore[reportArgumentType]
    else:
        wait_for_expr = ast.Constant(value=0)
    skip_sync_expr = ast.Constant(value=skip_sync)  # pyright: ignore[reportArgumentType]

    if wait_for is not None:
        call_triton_wait_signal = f"helion.runtime.triton_wait_{'' if is_scalar else 'multiple_'}signal(addr={bar_addrs}, expect=wait_for, update=signal, sem='{sem}', scope='{scope}', op='{op}', skip_sync=True, sync_before=(not skip_sync))"
        return expr_from_string(
            call_triton_wait_signal,
            signal_pad_arg=state.ast_arg(0) if as_ptrs else indices.index_expr,  # pyright: ignore[reportPossiblyUnboundVariable]
            wait_for=wait_for_expr,
            signal=signal_expr,
            skip_sync=skip_sync_expr,
        )
    call_triton_send_signal = f"helion.runtime.triton_send_signal(addr={bar_addrs}, update=signal, sem='{sem}', scope='{scope}', op='{op}', skip_sync=skip_sync)"

    return expr_from_string(
        call_triton_send_signal,
        signal_pad_arg=state.ast_arg(0) if as_ptrs else indices.index_expr,  # pyright: ignore[reportPossiblyUnboundVariable]
        signal=signal_expr,
        skip_sync=skip_sync_expr,
    )
