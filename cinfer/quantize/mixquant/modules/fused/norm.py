import torch
from torch import nn
import mixlib
 

class FasterTransformerRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6, cache = None):
        super().__init__()
        self.weight = weight.cuda().to(torch.float16)
        self.variance_epsilon = eps
        self.cache = cache
        self.next_layer = None

    @torch.no_grad()
    def forward(self, x):


        output = torch.empty_like(x)

        if self.next_layer is None:
            mixlib.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)

        else:
            if self.next_layer.bit == 8:
                self.cache.activation_outliers, self.cache.q_xcache = mixlib.layernorm_forward_cuda_extract_outliers(x, 
                self.weight, 
                output, self.variance_epsilon, 
                self.next_layer.ind, self.cache.x_scale)
            elif self.next_layer.bit == 4:
                self.cache.activation_outliers, self.cache.q_xcache = mixlib.layernorm_forward_cuda_extract_outliers_int4(x, 
                self.weight, 
                output, self.variance_epsilon, 
                self.next_layer.ind, self.cache.x_scale)


            else:
                raise NotImplementedError

        return output
