"""Which processor runs the ONNX models: the CPU, or a GPU the bundled runtime can drive.

The Windows installer bundles ONNX Runtime's DirectML build, which offers a
`DmlExecutionProvider` device for each hardware DirectX 12 GPU (NVIDIA, AMD or
Intel) through the ordinary display driver. macOS and the CPU wheel offer only the
CPU. A per-application GPU preference in Windows or a vendor control panel cannot
move this work: it picks the GPU that draws a program's graphics, while inference
runs in the backend process on the provider its session was created with.

A GPU is identified by its PCI identity, `gpu:<vendor>:<device>` in lowercase hex,
with `:<n>` for the nth identical card. DirectML addresses a GPU by its Windows
adapter index, which follows display wiring and the graphics preference, so the
index is never saved. ONNX Runtime reads its device list once per process, while
DirectML looks the index up again whenever it creates a session, so each request
matches the identity against Windows' current adapter list (`_dxgi_adapters`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import logging
import re
import uuid

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
DIRECTML = "DmlExecutionProvider"
NOT_FOUND = "The selected GPU was not found; running the models on the CPU"
# Windows' software renderer (WARP). DirectML refuses it by these ids whether or not DXGI
# flags it as software, and a GPU-less Windows runner lists it unflagged.
_BASIC_RENDER_DRIVER = (0x1414, 0x008C)
_GPU_ID = re.compile(r"gpu:[0-9a-f]{4,8}:[0-9a-f]{4,8}(?::(?:[2-9]|[1-9][0-9]))?")


@dataclass(frozen=True)
class Processor:
    id: str
    kind: str
    name: str
    adapter: int | None = None
    memory_mb: int | None = None
    provider: str = DIRECTML

    def public(self):
        return {"id": self.id, "kind": self.kind, "name": self.name, "memory_mb": self.memory_mb}


CPU = Processor("cpu", "cpu", "CPU")


def valid(processor) -> bool:
    return isinstance(processor, str) and (processor == CPU.id or re.fullmatch(r"cuda:GPU-[0-9a-fA-F-]+", processor) is not None or _GPU_ID.fullmatch(processor) is not None)


def _name(hardware, metadata):
    # ONNX Runtime replaces non-ASCII characters (a trademark sign) with spaces.
    name = " ".join(str(metadata.get("Description", "")).split())
    return name or " ".join(f"{hardware.vendor} GPU".split())


def _memory(metadata):
    value = str(metadata.get("DxgiVideoMemory", "")).removesuffix(" MB")
    return int(value) if value.isdigit() else None


def _identities(adapters):
    """{id: adapter index} for (index, vendor, device) triples, numbering identical cards in order."""
    ids, seen = {}, {}
    for index, vendor, device in sorted(adapters):
        identity = f"gpu:{vendor:04x}:{device:04x}"
        seen[identity] = seen.get(identity, 0) + 1
        ids[identity if seen[identity] == 1 else f"{identity}:{seen[identity]}"] = index
    return ids


def available() -> list[Processor]:
    """The CPU, then each GPU the bundled runtime can use, in Windows adapter order.

    This is ONNX Runtime's list, read when the backend first asks for it: a GPU added
    or given a driver later is offered after the app restarts.
    """
    if CUDA in ort.get_available_providers():
        from .gpu_batch import devices as cuda_devices
        found = cuda_devices()
        if found:
            return [CPU] + [Processor("cuda:"+d["uuid"], "gpu", d["name"], d["index"], d["total"]//(1024**2), CUDA) for d in found]
    try:
        devices = ort.get_ep_devices()
    except Exception:  # A runtime without device discovery can still use the CPU.
        devices = []
    try:
        gpus = {}
        for device in devices:
            hardware = device.device
            metadata = dict(hardware.metadata)
            adapter = str(metadata.get("DxgiAdapterNumber", ""))
            # DirectML silently uses adapter 0 for a device without an adapter number,
            # which may be a different GPU from the one named, so such devices are not offered.
            if (device.ep_name != DIRECTML or hardware.type != ort.OrtHardwareDeviceType.GPU or not adapter.isdigit()
                    or (hardware.vendor_id, hardware.device_id) == _BASIC_RENDER_DRIVER):
                continue
            gpus.setdefault(int(adapter), (hardware.vendor_id, hardware.device_id, _name(hardware, metadata), _memory(metadata)))
        ids = _identities((adapter, vendor, device) for adapter, (vendor, device, _, _) in gpus.items())
        return [CPU] + [Processor(identity, "gpu", gpus[adapter][2], adapter, gpus[adapter][3])
                        for identity, adapter in ids.items()]
    except Exception:
        logging.getLogger(__name__).warning("Could not read GPU device details", exc_info=True)
        return [CPU]


def resolve(requested: str) -> tuple[Processor, str | None]:
    """The processor a request runs on, and why it is not the one requested (None if it is)."""
    if requested == CPU.id:
        return CPU, None
    listed = next((processor for processor in available() if processor.id == requested), None)
    if listed is None:
        return CPU, NOT_FOUND
    if listed.provider == CUDA:
        return listed, None
    try:
        current = _identities(_dxgi_adapters())
    except Exception:
        # A stale adapter index can select an entirely different card.
        logging.getLogger(__name__).warning("Could not list Windows graphics adapters", exc_info=True)
        return CPU, "Could not verify the selected GPU identity; running the models on the CPU"
    if requested not in current:
        return CPU, NOT_FOUND
    return replace(listed, adapter=current[requested]), None


# Windows' adapter list through DXGI, the way DirectML's `device_id` indexes it:
# CreateDXGIFactory2(0, ...) then IDXGIFactory1::EnumAdapters1. Constants are from dxgi.h.
_DXGI_ERROR_NOT_FOUND = 0x887A0002 - (1 << 32)  # a negative HRESULT
_DXGI_ADAPTER_FLAG_REMOTE, _DXGI_ADAPTER_FLAG_SOFTWARE = 1, 2
_IID_IDXGI_FACTORY1 = uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le
_RELEASE, _GET_DESC1, _ENUM_ADAPTERS1 = 2, 10, 12  # vtable slots: IUnknown, IDXGIAdapter1, IDXGIFactory1
_HRESULT = ctypes.c_int32
# x64 Windows has one calling convention, so CFUNCTYPE serves where WINFUNCTYPE is absent.
_FUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


class _AdapterDesc1(ctypes.Structure):
    # DXGI_ADAPTER_DESC1. WCHAR is 16 bits on Windows; spelled so, the layout is the same
    # wherever the tests run. AdapterLuid is split into its two fields.
    _fields_ = [("Description", ctypes.c_uint16 * 128), ("VendorId", ctypes.c_uint32),
                ("DeviceId", ctypes.c_uint32), ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuidLowPart", ctypes.c_uint32),
                ("AdapterLuidHighPart", ctypes.c_int32), ("Flags", ctypes.c_uint32)]


def _method(interface, slot, *argtypes):
    """A COM method from the object's vtable, bound to the object."""
    vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    function = _FUNCTYPE(_HRESULT, ctypes.c_void_p, *argtypes)(vtable[slot])
    return lambda *args: function(interface, *args)


def _dxgi_adapters(create_factory=None):
    """(index, vendor, device) for each hardware adapter, skipping the software and remote ones as `available` does."""
    if create_factory is None:
        create_factory = ctypes.WinDLL("dxgi").CreateDXGIFactory2
        create_factory.restype = _HRESULT
    iid = (ctypes.c_ubyte * 16).from_buffer_copy(_IID_IDXGI_FACTORY1)
    factory = ctypes.c_void_p()
    result = create_factory(0, ctypes.byref(iid), ctypes.pointer(factory))
    if result < 0 or not factory.value:
        raise OSError(f"CreateDXGIFactory2 failed ({result & 0xFFFFFFFF:#010x})")
    adapters = []
    try:
        enum_adapters = _method(factory, _ENUM_ADAPTERS1, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            result = enum_adapters(index, ctypes.pointer(adapter))
            if result == _DXGI_ERROR_NOT_FOUND:
                return adapters
            if result < 0 or not adapter.value:
                raise OSError(f"EnumAdapters1({index}) failed ({result & 0xFFFFFFFF:#010x})")
            try:
                desc = _AdapterDesc1()
                if _method(adapter, _GET_DESC1, ctypes.POINTER(_AdapterDesc1))(ctypes.pointer(desc)) < 0:
                    raise OSError(f"GetDesc1 failed for adapter {index}")
                if (not desc.Flags & (_DXGI_ADAPTER_FLAG_REMOTE | _DXGI_ADAPTER_FLAG_SOFTWARE)
                        and (desc.VendorId, desc.DeviceId) != _BASIC_RENDER_DRIVER):
                    adapters.append((index, desc.VendorId, desc.DeviceId))
            finally:
                _method(adapter, _RELEASE)()
            index += 1
    finally:
        _method(factory, _RELEASE)()
