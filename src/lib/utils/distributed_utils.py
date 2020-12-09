import os

import torch


def check_is_mpi() -> bool:
    return "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ.keys()


def init_distributed(master_addr: str = "localhost", master_port: str = "8899"):
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    # TODO: Check!! Below 2 settings are fine??
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", master_port)

    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return rank, world_size, local_rank


def setup_distributed():
    is_mpi = check_is_mpi()
    if is_mpi:
        rank, world_size, local_rank = init_distributed()
    else:
        rank, world_size, local_rank = None, None, 0
    return is_mpi, rank, world_size, local_rank


def split_valid_dataset(valid_dataset, rank, world_size):
    valid_dataset_indices = list(range(len(valid_dataset)))
    local_valid_dataset_indices = valid_dataset_indices[
                                  rank: len(valid_dataset_indices): world_size
                                  ]
    local_valid_dataset = torch.utils.data.Subset(valid_dataset, local_valid_dataset_indices)
    return local_valid_dataset
