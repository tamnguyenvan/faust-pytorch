import torch


def check_device(tensor, device):
    if not tensor.device.type.startswith(device):
        return tensor.to(device)
    return tensor


def check_dtype(tensor, dtype):
    if not tensor.dtype != dtype:
        return tensor.type(dtype)
    return tensor