import torch
import torch.distributed as dist
import os
import argparse

os.environ["NCCL_DEBUG"] = "WARN"  # WARN, TRACE, INFO
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

def simple_nccl_test(rank, world_size, master_addr, master_port):
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank,
    )
    print(f"Rank {rank} initialized on {master_addr}:{master_port}.")

    # Assign tensor to the CUDA device associated with the rank
    tensor = torch.tensor([rank], dtype=torch.float32).to(f"cuda:{rank}")
    print(f"Rank {rank} before all_reduce has tensor value {tensor.item()}")

    # Perform all_reduce (sum all tensors across processes)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} after all_reduce has tensor value {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple NCCL test')
    parser.add_argument('--rank', type=int, required=True, help='Rank of the process')
    parser.add_argument('--world_size', type=int, default=2, help='Total number of processes')
    parser.add_argument('--master_addr', type=str, required=True, help='Master node IP address')
    parser.add_argument('--master_port', type=int, required=True, help='Master node port')

    args = parser.parse_args()
    rank = args.rank
    world_size = args.world_size
    master_addr = args.master_addr
    master_port = args.master_port

    # Print environment and versions
    print("PyTorch version:", torch.__version__)
    print("CUDA version detected by PyTorch:", torch.version.cuda)
    print("Is CUDA available?", torch.cuda.is_available())
    print("Is NCCL available?", torch.distributed.is_nccl_available())
    print("Using master address:", master_addr)
    print("Using master port:", master_port)

    simple_nccl_test(rank, world_size, master_addr, master_port)
