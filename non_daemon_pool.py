import time
from functools import partial

from torch import multiprocessing



class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        # self.data = init_fun(self.name) if init_fun is not None else None

    def run(self):
        proc_name = self.name
        while True:
            task_datum = self.task_queue.get()
            if task_datum is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            # print('%s: %s' % (proc_name, task))
            task, datum = task_datum
            answer = task(datum)
            self.task_queue.task_done()
            self.result_queue.put(answer)

class NonDaemonPool(object):

    def __init__(self, processes) -> None:
        super().__init__()
        self.n_jobs = processes
        # self.init_fun = init_fun
        self.task_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()


    def __enter__(self):
        consumers = [Consumer(self.task_queue,self.results_queue) for i in range(self.n_jobs)]
        for w in consumers:
            w.daemon = False
            w.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i in range(self.n_jobs):
            self.task_queue.put(None)
        self.task_queue.join()
        self.results_queue.close()

    def imap_unordered(self,fun, data_g):
        data_iter = iter(data_g)
        [self.task_queue.put((fun,next(data_iter))) for i in range(self.n_jobs)]
        for datum in data_iter:
            yield self.results_queue.get()
            self.task_queue.put((fun,datum))
        for i in range(self.n_jobs):
            yield self.results_queue.get()

def funfun(x):
    time.sleep(0.5)
    s = '%s: %d' % (str(multiprocessing.current_process()), x)
    print(s)
    return s

global_data = None
def fun(datum):
    global global_data
    if global_data is None:
        global_data = str(multiprocessing.current_process())

    with multiprocessing.Pool(processes=2) as p:
        result = list(p.imap_unordered(funfun,[datum+k for k in range(9)]))
    return {'global_data':global_data,'results':result}

if __name__ == '__main__':


    data = [10 * x for x in range(3)]
    with NonDaemonPool(processes=2) as p:
        x = list(p.imap_unordered(fun, data))
    print(len(x))
    print(x)
    assert len(x) == len(data)