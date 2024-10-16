import time
import inspect
import os
import pynvml
import torch
import numpy as np
from ..config import MetaData  

class ExecutionMonitor:
    def __init__(self, func):
        self.func = func
        self.meta_data = MetaData()  

    def __call__(self, *args, **kwargs):
        """Monitor GPU and execute the function"""
        # Get calling function and module info
        caller = inspect.getframeinfo(inspect.currentframe().f_back)
        module_name = os.path.basename(caller.filename)  
        function_name = self.func.__name__

        if len(args) > 0 and isinstance(args[0], object):
            instance = args[0]
            result = self.func(instance, *args[1:], **kwargs) 
        else:
            result = self.func(*args, **kwargs)

        gpu_memory_after = self.print_gpu_utilisation()

        title = f"{module_name}::{function_name}_gpu_utilisation"
        self.meta_data.update(title=gpu_memory_after)

        return result

    def print_gpu_utilisation(self):
        """Get GPU memory utilization using PyNVML"""
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            torch_gpu_id = torch.cuda.current_device()
            pynvml.nvmlInit() 
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            nvml_gpu_id = int(devices[torch_gpu_id]) 
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info_used = info.used // 1024 ** 2  # Convert to MB
            info_total = info.total // 1024 ** 2  # Convert to MB

            gpu_memory_info = {
                "used": info_used,
                "total": info_total,
                "percentage": np.round((info_used * 100) / info_total, 2)
            }

            print(f"GPU {nvml_gpu_id} memory occupied: {info_used}/{info_total} MB = {gpu_memory_info['percentage']}%.")

            pynvml.nvmlShutdown()

            return gpu_memory_info  
        else:
            print("CUDA_VISIBLE_DEVICES not set. Skipping GPU memory measurement.")
            return {"used": 0, "total": 0, "percentage": 0}
