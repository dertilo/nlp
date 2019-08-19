import multiprocessing
import subprocess
import torch

n_gpu = torch.cuda.device_count()

def queue_filler_fun(queue,nvidia_smi_process):
    def process_line(line):
        used,total = line.decode('utf-8').split('\n')[0].split(',')
        total = float(total)
        used = float(used)
        assert used<total
        return used / total
    while True:
        usages = [process_line(nvidia_smi_process.stdout.readline()) for k in range(n_gpu)]
        gpu_mem_usage = max(usages)
        queue.put(gpu_mem_usage)

class MonitorGpuMemUsage(object):
    nvidia_smi_process = None
    def __init__(self,interval:int) -> None:
        super().__init__()
        self.command = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l %d'%interval
        self.last_max = 1.0 # initially assume 100% usage

    def get_max_usage(self):
        l = []
        while True:
            try:
                l.append(self.queue.get(timeout=0.001))
            except Exception as e:
                break

        if len(l)>0:
            max_gpu_mem_usage = max(l); is_new = True
            self.last_max = max_gpu_mem_usage
        else:
            max_gpu_mem_usage = self.last_max; is_new=False

        return max_gpu_mem_usage,is_new

    def __enter__(self):
        self.nvidia_smi_process = subprocess.Popen(self.command.split(' '), stdout=subprocess.PIPE)

        self.queue = multiprocessing.Queue()
        self.queue_filler = multiprocessing.Process(target=queue_filler_fun, name='queue-filler',
                                                    args=(self.queue,self.nvidia_smi_process))
        self.queue_filler.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.nvidia_smi_process.terminate()
        self.nvidia_smi_process.kill()

        self.queue.cancel_join_thread()
        self.queue_filler.terminate()
