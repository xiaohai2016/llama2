#   Start the web server with the following command from within the llama 2 folder:
#       python web_serv.py
#   Prerequisite: pip install -r requirements.txt
import torch
from multiprocessing import Queue, Process, set_start_method

import http.server
import json
import os
from llama import Llama
import torch.distributed as dist
import re

#-----------------Configuration section------------------
PORT = 8000
MASTER_PORT = 5001
model_file = "../llama2-models/llama-2-13b-chat/"
tokenizer_path = "../llama2-models/tokenizer.model"
temperature = 0.3
max_gen_len = 2048
max_seq_len = 2048
max_batch_size = 6
version = int(re.compile(r'-(\d+)b-').search(model_file).group(1))
nproc_per_node = {7: 1, 13: 2, 70: 8}[version]
#--------------------------------------------------------

def inference(request_q, response_q, local_rank, world_size, temperature: float = 0.6, top_p: float = 0.9):
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    os.environ['NCCL_P2P_LEVEL']='LOC' # disable P2P: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-p2p-disable
    torch.set_num_threads(1)

    generator = Llama.build(
        ckpt_dir=model_file,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=world_size,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    to_quit = False
    buffer_size = max_seq_len * 10
    inp_t = torch.zeros(buffer_size, dtype=torch.int)
    while not to_quit:
        if local_rank == 0:
            inp_i = request_q.get()[:buffer_size - 1] # prevent overflow
            inp_t[:len(inp_i)] = torch.as_tensor([ord(ch) for ch in inp_i])
            inp_t[len(inp_i):] = 0
        dist.broadcast(inp_t, src=0)
        inp = ''.join([chr(u) for u in filter(lambda x: x != 0, inp_t.tolist())])
        if inp == "q" or inp == "x" or inp == "exit" or inp == "quit":
            to_quit = True
            print(f"Process {local_rank} exiting")
        if not to_quit:
            prompts = [inp]
            print(f"Got prompt: {inp}")
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                print_out=False,
            )
            if local_rank == 0:
                response_q.put(results[0]['generation'])

if __name__ == "__main__":

    print("---Starting the web server for LLAMA 2 models")
    print(f"    GPU count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"    Device {idx}: {torch.cuda.get_device_name(idx)}")
        print('         Memory Usage:')
        print('             Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('             Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    home_result = f"<p>A simple LLAMA 2 model server!</p><p>Model file: {model_file}</p>" + \
                f"<p>Token file: {tokenizer_path}</p><p>max_gen_len: {max_gen_len}</p>" + \
                    f"<p>max_seq_len: {max_seq_len}</p><p>nproc_per_node: {nproc_per_node}</p>"
    response_prefix = "<html><head><title>LLAMA 2 model HTTP server</title></head><body>".encode()
    response_postfix = "</body></head>\r\n".encode()

    set_start_method('spawn')
    request_queue = Queue()
    response_queue = Queue()

    class SimpleHttpHandler(http.server.BaseHTTPRequestHandler):
        def _set_headers(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
        def do_HEAD(self):
            self._set_headers()
        def do_GET(self):
            if self.path != "/":
                self.send_response(404)
                self.end_headers
            else:
                self._set_headers()
                self.wfile.write(response_prefix)
                self.wfile.write(home_result.encode())
                self.wfile.write(response_postfix)
        def do_POST(self):
            if self.path != "/":
                self.send_response(404)
                self.end_headers
            else:
                content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
                post_data = self.rfile.read(content_length).decode("utf-8") # <--- Gets the data itself
                posted_content = json.loads(post_data)

                # Process the request
                if_exit = False
                if posted_content['user'] == 'exit' or posted_content['user'] == 'quit':
                    if_exit = True
                request_queue.put(posted_content['user'])
                if not if_exit:
                    response = response_queue.get()
                else:
                    response = ""

                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                if not if_exit:
                    self.wfile.write(json.dumps({"message": response}).encode())
                self.wfile.write("\r\n".encode())

    server = http.server.HTTPServer(('localhost', PORT), SimpleHttpHandler)
    print(f'Started http server at http://localhost:{PORT}')

    # runs nproc_per_node processes
    processes = []
    for idx in range(nproc_per_node):
        p = Process(target=inference, args=(request_queue, response_queue, idx, nproc_per_node, temperature,))
        p.start()
        processes.append(p)

    server.serve_forever()

    print("Received instruction to shut down!")

    # waits for all copies of inferencers to finish
    for p in processes:
        p.join()