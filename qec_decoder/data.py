"""Data generation and input mapping utilities for surface-code decoding."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import scipy.sparse
import stim
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset


class QECDataset(Dataset):
    """Loads pre-generated detector events and logical labels from an ``.npz`` file."""

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        with np.load(file_path) as data:
            self.inputs = data["X"]
            self.targets = data["Y"]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.inputs[idx]).float(), torch.from_numpy(self.targets[idx]).float()

    def get_input_dim(self) -> int:
        return self.inputs.shape[1]


def make_surface_code_circuit(distance: int, rounds: Optional[int] = None, p: float = 0.005) -> stim.Circuit:
    """Creates the rotated memory-Z surface-code circuit used in the paper."""

    if rounds is None:
        rounds = distance
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


def generate_surface_code_data(distance: int, rounds: int, noise_rate: float, num_shots: int):
    """Samples detector events and observable flips from a Stim surface-code circuit."""

    circuit = make_surface_code_circuit(distance, rounds, noise_rate)
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(shots=num_shots, separate_observables=True)
    dem = circuit.detector_error_model(decompose_errors=True)
    return circuit, dem, detection_events, observable_flips


def create_benchmark_data(
    d: int,
    r: Optional[int] = None,
    p: float = 0.005,
    train_shots: int = 100_000,
    test_shots: int = 50_000,
    save_dir: str = "data",
) -> None:
    """Creates train/test ``.npz`` files and the matching detector error model."""

    if r is None:
        r = d
    os.makedirs(save_dir, exist_ok=True)

    circuit, dem, x_train, y_train = generate_surface_code_data(d, r, p, train_shots)
    np.savez_compressed(
        os.path.join(save_dir, f"train_{d}_{r}_{p}.npz"),
        X=x_train.astype(np.uint8),
        Y=y_train.astype(np.uint8),
    )
    dem.to_file(os.path.join(save_dir, f"dem_{d}_{r}_{p}.dem"))

    _, _, x_test, y_test = generate_surface_code_data(d, r, p, test_shots)
    np.savez_compressed(
        os.path.join(save_dir, f"test_{d}_{r}_{p}.npz"),
        X=x_test.astype(np.uint8),
        Y=y_test.astype(np.uint8),
    )


class OnlineSurfaceCodeDataset(IterableDataset):
    """Infinite online dataset sampled directly from Stim."""

    def __init__(self, d: int, r: Optional[int] = None, p: float = 0.005, batch_size: int = 1024):
        self.d = d
        self.r = d if r is None else r
        self.p = p
        self.internal_batch_size = batch_size
        self.circuit = make_surface_code_circuit(d, self.r, p)
        self.sampler = self.circuit.compile_detector_sampler()
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.id + int(torch.initial_seed() % 2**32))

        while True:
            batch = self.sampler.sample(
                shots=self.internal_batch_size,
                append_observables=True,
            ).astype(np.float32)
            detectors = batch[:, : self.num_detectors]
            observables = batch[:, self.num_detectors :]
            for i in range(self.internal_batch_size):
                yield detectors[i], observables[i]

    def get_input_output_dim(self) -> tuple[int, int]:
        return self.num_detectors, self.num_observables

    def get_circuit(self) -> stim.Circuit:
        return self.circuit

    def get_dem(self) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(decompose_errors=True)


class StimTo3DMapper(nn.Module):
    """Maps a flat Stim detector-event vector to ``(B, 2, T, H, W)``."""

    def __init__(self, circuit: stim.Circuit, d: int, r: int):
        super().__init__()
        self.d = d
        self.r = r

        coords_dict = circuit.get_detector_coordinates()
        ts, xs, ys = [], [], []
        for k in sorted(coords_dict):
            coord = coords_dict[k]
            xs.append(coord[0])
            ys.append(coord[1])
            ts.append(coord[2] if len(coord) == 3 else r - 1)

        unique_ts = sorted(set(ts))
        t_map = {val: i for i, val in enumerate(unique_ts)}
        min_x, min_y = min(xs), min(ys)

        map_t, map_h, map_w, map_c = [], [], [], []
        for k in range(circuit.num_detectors):
            coord = coords_dict.get(k)
            if coord is None:
                map_t.append(0)
                map_h.append(0)
                map_w.append(0)
                map_c.append(0)
                continue

            x, y = coord[0], coord[1]
            t = coord[2] if len(coord) == 3 else r - 1
            map_c.append(0 if (int(x) + int(y)) % 4 == 0 else 1)
            map_t.append(t_map[t])
            map_h.append(int((y - min_y) // 2))
            map_w.append(int((x - min_x) // 2))

        self.register_buffer("map_t", torch.tensor(map_t, dtype=torch.long))
        self.register_buffer("map_h", torch.tensor(map_h, dtype=torch.long))
        self.register_buffer("map_w", torch.tensor(map_w, dtype=torch.long))
        self.register_buffer("map_c", torch.tensor(map_c, dtype=torch.long))
        self.max_t = len(unique_ts)
        self.max_h = max(map_h) + 1
        self.max_w = max(map_w) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        volume = torch.zeros(
            x.shape[0],
            2,
            self.max_t,
            self.max_h,
            self.max_w,
            device=x.device,
            dtype=x.dtype,
        )
        volume[:, self.map_c, self.map_t, self.map_h, self.map_w] = x
        return volume


class StimToTemporalGridMapper(nn.Module):
    """Maps detector events to ``(B, T, C, H, W)`` with a fixed detector embedding."""

    def __init__(self, circuit: stim.Circuit, d: int, r: int, channels: int, seed: int = 0):
        super().__init__()
        self.d = d
        self.r = r
        self.channels = channels

        coords_dict = circuit.get_detector_coordinates()
        ts, xs, ys = [], [], []
        for k in sorted(coords_dict):
            coord = coords_dict[k]
            xs.append(coord[0])
            ys.append(coord[1])
            ts.append(coord[2] if len(coord) == 3 else r - 1)

        unique_ts = sorted(set(ts))
        t_map = {val: i for i, val in enumerate(unique_ts)}
        min_x, min_y = min(xs), min(ys)

        map_t, map_h, map_w = [], [], []
        for k in range(circuit.num_detectors):
            coord = coords_dict.get(k)
            if coord is None:
                map_t.append(0)
                map_h.append(0)
                map_w.append(0)
                continue
            x, y = coord[0], coord[1]
            t = coord[2] if len(coord) == 3 else r - 1
            map_t.append(t_map[t])
            map_h.append(int((y - min_y) // 2))
            map_w.append(int((x - min_x) // 2))

        self.max_t = len(unique_ts)
        self.max_h = max(map_h) + 1
        self.max_w = max(map_w) + 1
        flat_index = (
            torch.tensor(map_t, dtype=torch.long) * self.max_h * self.max_w
            + torch.tensor(map_h, dtype=torch.long) * self.max_w
            + torch.tensor(map_w, dtype=torch.long)
        )
        self.register_buffer("flat_index", flat_index)

        generator = torch.Generator()
        generator.manual_seed(seed)
        embedding = torch.randn(circuit.num_detectors, channels, generator=generator)
        embedding = embedding / max(channels, 1) ** 0.5
        self.register_buffer("detector_embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.to(self.detector_embedding.dtype).unsqueeze(-1) * self.detector_embedding.unsqueeze(0)
        flat = torch.zeros(
            x.shape[0],
            self.max_t * self.max_h * self.max_w,
            self.channels,
            device=x.device,
            dtype=features.dtype,
        )
        index = self.flat_index.view(1, -1, 1).expand(x.shape[0], -1, self.channels)
        flat.scatter_add_(1, index, features)
        return flat.view(x.shape[0], self.max_t, self.max_h, self.max_w, self.channels).permute(0, 1, 4, 2, 3)


def native_scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: Optional[int] = None):
    """Small ``scatter_add`` wrapper implemented with native PyTorch."""

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    if index.dim() < src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index = index.view(*view_shape).expand_as(src)
    return out.scatter_add_(dim, index, src)


class StimToGraphMapper(nn.Module):
    """Converts a Stim detector error model into batched Tanner-graph tensors."""

    def __init__(self, dem: stim.DetectorErrorModel):
        super().__init__()
        rows_v, cols_c, error_probs = [], [], []
        var_idx = 0
        for instruction in dem:
            if instruction.type != "error":
                continue
            p = instruction.args_copy()[0]
            dets = [
                t.val
                for t in instruction.targets_copy()
                if not t.is_logical_observable_id() and not t.is_separator()
            ]
            if dets:
                for det_id in dets:
                    rows_v.append(var_idx)
                    cols_c.append(det_id)
                error_probs.append(p)
                var_idx += 1

        self.num_checks = dem.num_detectors
        self.num_vars = var_idx
        safe_probs = np.clip(error_probs, 1e-10, 1.0 - 1e-10)
        initial_llrs = np.log((1 - safe_probs) / safe_probs)

        self.register_buffer("base_v2c_src", torch.tensor(rows_v, dtype=torch.long))
        self.register_buffer("base_v2c_dst", torch.tensor(cols_c, dtype=torch.long))
        self.register_buffer("base_c2v_src", torch.tensor(cols_c, dtype=torch.long))
        self.register_buffer("base_c2v_dst", torch.tensor(rows_v, dtype=torch.long))
        self.register_buffer("initial_llrs", torch.tensor(initial_llrs, dtype=torch.float32).unsqueeze(1))

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        device = x.device
        x_c_flat = x.reshape(-1, 1)
        x_v_flat = self.initial_llrs.repeat(batch_size, 1)

        num_edges = self.base_v2c_src.numel()
        batch_ids = torch.arange(batch_size, device=device).repeat_interleave(num_edges)
        c_offsets = batch_ids * self.num_checks
        v_offsets = batch_ids * self.num_vars

        v2c_src = self.base_v2c_src.repeat(batch_size) + v_offsets
        v2c_dst = self.base_v2c_dst.repeat(batch_size) + c_offsets
        c2v_src = self.base_c2v_src.repeat(batch_size) + c_offsets
        c2v_dst = self.base_c2v_dst.repeat(batch_size) + v_offsets

        return (
            x_c_flat,
            x_v_flat,
            torch.stack([v2c_src, v2c_dst], dim=0),
            torch.stack([c2v_src, c2v_dst], dim=0),
            torch.arange(batch_size, device=device).repeat_interleave(self.num_vars),
        )


class SyndromeConsistencyLoss(nn.Module):
    """Soft-XOR syndrome consistency loss used by the neural BP decoder."""

    def forward(self, logits: torch.Tensor, syndrome_targets: torch.Tensor, edge_index_v2c: torch.Tensor):
        probs = torch.sigmoid(logits).view(-1)
        term = torch.clamp(1.0 - 2.0 * probs, min=-0.9999, max=0.9999)
        log_abs = torch.log(torch.clamp(torch.abs(term), min=1e-6))
        sign = torch.sign(term)

        check_sum_log = native_scatter_add(log_abs[edge_index_v2c[0]], edge_index_v2c[1], dim=0)
        neg_count = native_scatter_add((sign < 0).float()[edge_index_v2c[0]], edge_index_v2c[1], dim=0)
        pred_s = 0.5 * (1.0 - ((-1.0) ** torch.round(neg_count)) * torch.exp(check_sum_log))
        return nn.functional.mse_loss(pred_s, syndrome_targets.view(-1))


def dem_to_check_matrix(dem: stim.DetectorErrorModel):
    """Converts a detector error model into sparse check/logical matrices."""

    h_rows, h_cols, l_rows, l_cols, probs = [], [], [], [], []
    var_idx = 0
    for instruction in dem:
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        dets, obs_ids = [], []
        for target in instruction.targets_copy():
            if target.is_logical_observable_id():
                obs_ids.append(target.val)
            elif not target.is_separator():
                dets.append(target.val)
        if dets:
            h_rows.extend(dets)
            h_cols.extend([var_idx] * len(dets))
            l_rows.extend(obs_ids)
            l_cols.extend([var_idx] * len(obs_ids))
            probs.append(p)
            var_idx += 1

    if dem.num_detectors == 0 or var_idx == 0:
        raise ValueError(f"Empty check matrix: detectors={dem.num_detectors}, variables={var_idx}")

    h_data = np.ones(len(h_rows), dtype=np.uint8)
    l_data = np.ones(len(l_rows), dtype=np.uint8)
    h = scipy.sparse.csr_matrix((h_data, (h_rows, h_cols)), shape=(dem.num_detectors, var_idx))
    l = scipy.sparse.csr_matrix((l_data, (l_rows, l_cols)), shape=(dem.num_observables, var_idx))
    return h, l, np.asarray(probs)
