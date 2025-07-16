from __future__ import annotations

import torch
from torch._C._distributed_c10d import DeviceType
from torch._C._distributed_c10d import Store
from torch._C._distributed_c10d import _SymmetricMemory


class FakeSymmetricMemory(_SymmetricMemory):
    def __init__(self, rank: int = 0, world_size: int = 1):
        self._rank = rank
        self._world_size = world_size
        self._buffer_size = 0
        self._signal_pad_size = 0

    @staticmethod
    def set_group_info(
        group_name: str,
        rank: int,
        world_size: int,
        store: Store,
    ) -> None:
        pass

    @staticmethod
    def empty_strided_p2p(
        size: torch.types._size,
        stride: torch.types._size,
        dtype: torch.dtype,
        device: torch.device,
        group_name: str | None = None,
        alloc_id: int | None = None,
    ) -> torch.Tensor:
        return torch.empty_strided(size, stride, dtype=dtype, device=device)

    @staticmethod
    def has_multicast_support(
        device_type: DeviceType,
        device_idx: int,
    ) -> bool:
        return False

    @staticmethod
    def set_backend(name: str) -> None:
        pass

    @staticmethod
    def get_backend(device: torch.device) -> str | None:
        return None

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @staticmethod
    def rendezvous(
        tensor: torch.Tensor, group_name: str | None = None
    ) -> FakeSymmetricMemory:
        return FakeSymmetricMemory()

    def get_buffer(
        self,
        rank: int,
        sizes: torch.types._size,
        dtype: torch.dtype,
        storage_offset: int | None = 0,
    ) -> torch.Tensor:
        return torch.zeros(sizes, dtype=dtype)

    def get_signal_pad(
        self,
        rank: int,
        sizes: torch.types._size = [],
        dtype: torch.dtype | None = None,
        storage_offset: int | None = 0,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.int32
        if not sizes:
            sizes = (1,)
        return torch.zeros(sizes, dtype=dtype)

    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None:
        pass

    def put_signal(
        self,
        dst_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None:
        pass

    def wait_signal(
        self,
        src_rank: int,
        channel: int = 0,
        timeout_ms: int = 0,
    ) -> None:
        pass

    @staticmethod
    def memset32(
        tensor: torch.Tensor, offset: int, val: int, count: int = 1
    ) -> torch.Tensor:
        return tensor

    @staticmethod
    def stream_write_value32(
        tensor: torch.Tensor, offset: int, val: int
    ) -> torch.Tensor:
        return tensor

    @property
    def buffer_ptrs(self) -> list[int]:
        return self._buffer_ptrs

    @property
    def buffer_ptrs_dev(self) -> int:
        return 0

    @property
    def signal_pad_ptrs(self) -> list[int]:
        return self._signal_pad_ptrs

    @property
    def signal_pad_ptrs_dev(self) -> int:
        return 0

    @property
    def multicast_ptr(self) -> int:
        return 0

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def signal_pad_size(self) -> int:
        return self._signal_pad_size

    def __getitem__(self, index):
        return self._data.get(index, 0)

    def __setitem__(self, index, value):
        self._data[index] = value

    def __len__(self):
        return len(self._data)
