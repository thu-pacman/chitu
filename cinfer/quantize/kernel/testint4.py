import torch

M = 32
K = 128
N = 256
a = torch.randn((M,K),dtype=torch.float16,device='cuda')
weight = torch.randn((N,K),dtype=torch.float16,device='cuda')



from EETQ import quant_weights, preprocess_weights, w4_a16_gemm

int4_weight_cpu = torch.t(weight).contiguous().cpu()
int4_weight, scales = quant_weights(int4_weight_cpu, torch.int4, False)

#print(scales)
 
qweight = int4_weight.to(weight.device)
weight_scales = scales.half().to(weight.device)
 
print(weight_scales)
grand = torch.mm(a,weight.T)
import sys
sys.path.append('/home/chenyidong/quant/cutlass_quant/build/lib.linux-x86_64-cpython-310')

y =   w4_a16_gemm(a, qweight, weight_scales)
print(y)
print(grand)