from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(x, device) for x in value)
    if isinstance(value, list):
        return [_move_to_device(x, device) for x in value]
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _call(module_or_function: Callable, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return module_or_function(**inputs)
    if isinstance(inputs, tuple):
        return module_or_function(*inputs)
    return module_or_function(inputs)


def count_gflops(
    model: torch.nn.Module,
    inputs: Any,
) -> tuple[float, dict[str, int]]:
    """
    Count FLOPs for one forward pass.

    Returns:
        gflops
        unsupported operations reported by fvcore
    """
    model.eval()

    analysis = FlopCountAnalysis(model, inputs)
    total_flops = float(analysis.total())

    unsupported = {
        str(name): int(count)
        for name, count in analysis.unsupported_ops().items()
    }

    return total_flops / 1e9, unsupported


@torch.inference_mode()
def measure_latency_ms(
    function: Callable[[], Any],
    device: torch.device,
    warmup: int = 10,
    repetitions: int = 50,
) -> tuple[float, float]:
    """
    Measure mean and standard-deviation latency in milliseconds.
    """
    for _ in range(warmup):
        function()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms = []

    for _ in range(repetitions):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            function()
            end.record()

            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            function()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        times_ms.append(elapsed_ms)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def profile_pipeline(
    *,
    project: str,
    model_name: str,
    core_model: torch.nn.Module,
    core_inputs: Any,
    frontend_function: Optional[Callable[[], Any]] = None,
    end_to_end_function: Optional[Callable[[], Any]] = None,
    frontend_model: Optional[torch.nn.Module] = None,
    frontend_inputs: Any = None,
    output_csv: str = "compute_results.csv",
    device: str = "cuda",
    warmup: int = 10,
    repetitions: int = 50,
) -> dict[str, Any]:
    """
    Profile one 3-second sample with batch size 1.

    Measures:
      - core model GFLOPs
      - frontend GFLOPs, when the frontend is a traceable PyTorch model
      - core latency
      - frontend latency
      - end-to-end latency
      - parameter count
    """
    device_obj = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    core_model = core_model.to(device_obj).eval()
    core_inputs = _move_to_device(core_inputs, device_obj)

    core_gflops, core_unsupported = count_gflops(
        core_model,
        core_inputs,
    )

    core_mean_ms, core_std_ms = measure_latency_ms(
        lambda: _call(core_model, core_inputs),
        device=device_obj,
        warmup=warmup,
        repetitions=repetitions,
    )

    frontend_gflops = None
    frontend_unsupported = {}

    if frontend_model is not None and frontend_inputs is not None:
        frontend_model = frontend_model.to(device_obj).eval()
        frontend_inputs = _move_to_device(frontend_inputs, device_obj)

        frontend_gflops, frontend_unsupported = count_gflops(
            frontend_model,
            frontend_inputs,
        )

    frontend_mean_ms = None
    frontend_std_ms = None

    if frontend_function is not None:
        frontend_mean_ms, frontend_std_ms = measure_latency_ms(
            frontend_function,
            device=device_obj,
            warmup=warmup,
            repetitions=repetitions,
        )

    end_to_end_mean_ms = None
    end_to_end_std_ms = None

    if end_to_end_function is not None:
        end_to_end_mean_ms, end_to_end_std_ms = measure_latency_ms(
            end_to_end_function,
            device=device_obj,
            warmup=warmup,
            repetitions=repetitions,
        )

    result = {
        "project": project,
        "model": model_name,
        "core_gflops": core_gflops,
        "frontend_gflops": frontend_gflops,
        "total_measurable_gflops": (
            core_gflops + frontend_gflops
            if frontend_gflops is not None
            else core_gflops
        ),
        "core_latency_ms": core_mean_ms,
        "core_latency_std_ms": core_std_ms,
        "frontend_latency_ms": frontend_mean_ms,
        "frontend_latency_std_ms": frontend_std_ms,
        "end_to_end_latency_ms": end_to_end_mean_ms,
        "end_to_end_latency_std_ms": end_to_end_std_ms,
        "parameters": sum(p.numel() for p in core_model.parameters()),
        "core_unsupported_ops": str(core_unsupported),
        "frontend_unsupported_ops": str(frontend_unsupported),
        "device": str(device_obj),
        "batch_size": 1,
    }

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    return result


class AudioOnlyWrapper(torch.nn.Module):
    """
    Generic wrapper for models that take masked mel only.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, masked_mel):
        output = self.model(masked_mel)

        if isinstance(output, (tuple, list)):
            return output[0]

        return output


class TwoInputAVWrapper(torch.nn.Module):
    """
    Generic wrapper for models that take:
        masked_mel, visual_features

    Reverse the argument order here if a model expects visual features first.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, masked_mel, visual_features):
        output = self.model(masked_mel, visual_features)

        if isinstance(output, (tuple, list)):
            return output[0]

        return output


class AVPLCWrapper(torch.nn.Module):
    """
    Wrapper for AV_PLC models returning:
        fused_spec, audio_spec, video_spec, ...
    """

    def __init__(self, model: torch.nn.Module, mode: str = "av"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(
        self,
        frames,
        speaker_embedding,
        masked_mel,
        audio_length,
        availability,
    ):
        output = self.model(
            frames,
            speaker_embedding,
            masked_mel,
            audio_length,
            availability,
        )

        fused_spec, audio_spec, video_spec = output[:3]

        if self.mode == "a":
            return audio_spec
        if self.mode == "v":
            return video_spec

        return fused_spec


if __name__ == "__main__":
    print(
        "Import profile_pipeline and one of the wrappers into each project runner.\n"
        "Use a real batch with batch size 1 and FP32 inference."
    )
