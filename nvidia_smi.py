import torch

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")
        print(f"    Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9} GB")
        print(f"    Memory Cached: {torch.cuda.memory_reserved(i) / 1e9} GB")
