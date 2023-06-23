from typing import Tuple

from time import perf_counter

import numpy as np

import torch
assert torch.cuda.is_available(), "This script requires a GPU."

from transformers import Pipeline


def get_speed_benchmark(pipeline: Pipeline,
                        query: dict,
                        num_warmup: int,
                        num_timed_runs: int,
                        num_beams: int) -> Tuple[float, float]:
    """
    Get the speed benchmark `(time_avg_ms, time_std_ms)` for the pipeline.
    """
    
    # Get placeholder for latencies:
    latencies = []
    
    # Important note:
    # The pipeline consummes the input, so we need to make a copy of the query for each run.
    
    # Warmup:
    print(f"Warming up for {num_warmup} runs...")
    for _ in range(num_warmup):
        _ = pipeline(query.copy(), generate_kwargs={"num_beams": num_beams})  # type: ignore
    
    # Timed run
    print(f"Running for {num_timed_runs} timed runs...")
    for _ in range(num_timed_runs):
        start_time = perf_counter()
        _ = pipeline(query.copy(), generate_kwargs={"num_beams": num_beams})  # type: ignore
        latency = perf_counter() - start_time
        latencies.append(latency)
    
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
        
    return time_avg_ms, time_std_ms
