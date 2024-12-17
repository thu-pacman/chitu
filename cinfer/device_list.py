import torch


class DeviceList:
    """Amortized O(1) appendable list on device"""

    def __init__(self, data=[], dtype=None, device=None):
        self._data = torch.tensor(data, dtype=dtype, device=device)
        self._len = len(data)

    def __len__(self):
        return self._len

    def append(self, item):
        if self._len == len(self._data):
            new_data = torch.empty(
                max(2 * self._len, 32), dtype=self._data.dtype, device=self._data.device
            )
            new_data[: self._len] = self._data
            self._data = new_data
        self._data[self._len] = item
        self._len += 1

    def to_tensor(self):
        return self._data[: self._len]

    def __getitem__(self, idx):
        return self.to_tensor()[idx]
