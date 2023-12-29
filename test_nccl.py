#   Start the web server with the following command from within the llama 2 folder:
#       python web_serv.py
#   Prerequisite: pip install -r requirements.txt
import torch
from multiprocessing import Queue, Process, set_start_method

import os
import torch.distributed as dist
import fairscale

#-----------------Configuration section------------------
MASTER_PORT = 5003
nproc_per_node = 2
#--------------------------------------------------------

def inference(request_q, response_q, local_rank, world_size, temperature: float = 0.6, top_p: float = 0.9):
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    os.environ['NCCL_P2P_LEVEL']='LOC' # disable P2P: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-disable

    torch.set_num_threads(1)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        device = torch.device('cuda')
    else:
        torch.distributed.init_process_group("gloo")
        device = torch.device('cpu')
    fairscale.nn.model_parallel.initialize.initialize_model_parallel(world_size)
    inp_t = torch.zeros(2048, dtype=torch.int).to(device)
    dist.broadcast(inp_t, src=0)
    print(f"Local rank: {local_rank} done!")

if __name__ == "__main__":

    print("Main process started")
    set_start_method('spawn')
    request_queue = Queue()
    response_queue = Queue()

    # runs nproc_per_node processes
    print("about to spawn new processes")
    processes = []
    for idx in range(nproc_per_node):
        p = Process(target=inference, args=(request_queue, response_queue, idx, nproc_per_node,))
        p.start()
        processes.append(p)

    # waits for all copies of inferencers to finish
    for p in processes:
        p.join()
    
    print("all sub processes should be done by now.")