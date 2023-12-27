# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List
import os

import torch.distributed as dist
import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    to_quit = False
    inp_t = torch.zeros(max_seq_len, dtype=torch.int)
    inp_t.share_memory_()
    while not to_quit:
        if local_rank == 0:
            print(f"\n{bcolors.OKCYAN}### Question:{bcolors.ENDC} ", end='')
            inp_i = input()
            inp_t[:len(inp_i)] = torch.as_tensor([ord(ch) for ch in inp_i])
            inp_t[len(inp_i):] = 0
        dist.barrier()
        inp = ''.join([chr(u) for u in filter(lambda x: x != 0, inp_t.tolist())])
        if inp == "q" or inp == "x" or inp == "exit" or inp == "quit":
            to_quit = True
        if not to_quit:
            prompts = [inp]
            if local_rank == 0:
                print(f"\n{bcolors.OKCYAN}### Answer:{bcolors.ENDC} ", end="")
            generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                print_out=True,
            )
            if local_rank == 0:
                print(
                    f"\n\n  Input token speed: {bcolors.OKGREEN}{generator.in_toks_ps} tok/s{bcolors.ENDC},",
                    f" Output token speed: {bcolors.OKCYAN}{generator.out_toks_ps} tok/s{bcolors.ENDC}."
                )



if __name__ == "__main__":
    fire.Fire(main)
