
import torch
 
@codegen(dtype = [torch.float16, torch.int8 , torch.int32], codegen = "cutlass")
def mixgemm(a, bint8, ind):
    afp = a[:,ind]
    bfp = bint8[:,ind].to(torch.float16).scale()
    a[:,ind] = 0
    aint8 = a.to(torch.int8)
    c = (aint8 * bint8).to(torch.float16).scale() + afp * bfp
    out = torch.relu(c)
    return out 



class Attention:
    def __init__(self,weight,oproj):
        self.qweight = weight
        self.oproj = oproj


    @quant([function=self.qweight, type="Mix",activation="A8",weight="W8",codegen = "cutlass"], 
           [function=self.oproj, type="Weight",activation="A16",weight="W8",codegen = "cutlass"])
    def attention(self,hidden_state):
        
        qx = self.qweight(hidden_state)
        o = self.oproj(qx)

        return o