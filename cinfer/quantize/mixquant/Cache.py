import torch



class MixLibCache:
    def __init__(self, inputdim = 1024, sigma = 6, bit = 8, eval_ppl = False, locality = False):
        self.device = 'cuda'
        self.x_scale = torch.zeros((inputdim,1),dtype=torch.float16).to('cuda')

        self.sigma = torch.zeros((1,1),dtype=torch.float16).to('cuda')  
        self.zeros = torch.zeros((inputdim,12288*3),dtype=torch.float16).to('cuda')    
        self.sigma[0] = sigma
        

        self.ind = None
        self.shape = None
        self.activation_outliers = None
        self.is_prefill = False
        self.bit = bit

        self.max_outliers = 256
        self.stop = 2

        self.eval_ppl = eval_ppl
        self.locality = locality
    def do_bench_cudagraph(self, fn):
        if torch.cuda.current_stream() == torch.cuda.default_stream():
            raise RuntimeError("Cannot capture graph in default stream. Please use side stream in benchmark code.")
        # warmup
        for i in range(10):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()


        return g
    


class MLPCache:
    def __init__(self, max_batch_size = 4096):
        self.device = 'cuda'
        self.x_scale = torch.zeros((max_batch_size,1),dtype=torch.float16).to('cuda')
        self.ind = None
        self.shape = None
        self.activation_outliers = None

