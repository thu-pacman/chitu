import torch

M = 32
K = 128
N = 256
a = torch.randn((M,K),dtype=torch.float16,device='cuda')
weight = torch.randn((N,K),dtype=torch.float16,device='cuda')



from EETQ import quant_weights, preprocess_weights, w8_a16_gemm_

int8_weight_cpu = torch.t(weight).contiguous().cpu()
int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)

#print(scales)
print(weight.amax(dim=1).cpu()/scales)
 
qweight = int8_weight.to(weight.device)
weight_scales = scales.half().to(weight.device)
 
print(weight_scales)
grand = torch.mm(a,weight.T)
import sys
sys.path.append('/home/chenyidong/quant/cutlass_quant/build/lib.linux-x86_64-cpython-310')

y = torch.ones((M,N),dtype=torch.float16,device='cuda')*100
w8_a16_gemm_(a, qweight, weight_scales,y,M,N,K)
print(y)
print(grand)