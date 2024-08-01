 
import torch
import torch.nn as nn
import sys
#import mixlib

 
 
from EETQ import quant_weights, preprocess_weights, w8_a16_gemm

from torch import Tensor
def two_compl(x: Tensor, bits: int) -> Tensor:
    return torch.where(x < 0, 2 ** bits + x, x)
def pack_to_i4(X: Tensor):

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4

def unpack_int8_to_int4(weight,ind):
    assert weight.dim() == 2
    return mixlib.unpack_int4_to_fp16(weight,ind)
 


class MixLinear_GEMM(nn.Module):
    def __init__(self, in_features, out_features, bias, dev,  bit, 
            weight_only = False, cache = None, fp_features_num = 128, name = None):
        super().__init__()
        
 
        self.in_features = in_features
        self.out_features = out_features
        self.bit = bit
 


         

        self.register_buffer('q_weight', torch.empty((in_features,out_features), dtype=torch.int8, device=dev,requires_grad=False))
        self.register_buffer('scale_col', torch.empty((out_features), dtype=torch.float16, device=dev,requires_grad=False))
    

        if bias:

            self.register_buffer('bias', torch.empty((out_features), dtype=torch.float16, device=dev,requires_grad=False))
        else:
            self.bias = None
        self.cnt = 0
        self.forward_without_precondition_len = -1
        if bit == 4:
            self.forward_without_precondition_len = fp_features_num


        self.cache = cache
        self.weight_only = weight_only


        self.add_outliers = True

         
        if cache is not None:
            self.sigma = torch.ones((1, 1),dtype=torch.float16, requires_grad=False,
                                            device = dev)
            self.sigma[0] = cache.sigma

        self.arch = torch.cuda.get_device_capability()[0]
        self.name = name

    @classmethod
    def from_linear(cls, linear, bit, weight_only=False, init_only=False,cache=None, 
                    layer_scales= None, dev = 'cuda', name = None, fp_features_num = 128):


        quant_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, 
                           dev, bit=bit, weight_only=weight_only,
                           cache=cache, name = name, fp_features_num = fp_features_num)
   

        if init_only is True: 
            return quant_linear   
             
        if weight_only is True:
            int8_weight_cpu = torch.t(linear.weight.data).contiguous().cpu()
            int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)
            quant_linear.q_weight.copy_ (int8_weight)
            quant_linear.scale_col.copy_(scales.half())

        else:


            if bit == 8:

                scale =   (torch.max(torch.abs(linear.weight.data), dim=1)[0].unsqueeze(1) / (
                                127)).to(torch.float16).reshape((1,linear.out_features))
                quant_linear.scale_col.copy_(scale)
                tmp = linear.weight.data.cuda()
                tmp /= quant_linear.scale_col.T
                tmp = tmp.round().to(torch.int8)
                quant_linear.q_weight.copy_(tmp)   
            else:


                assert layer_scales is not None
                fp_features = quant_linear.fp_features_num
                linear.ind = torch.sort(layer_scales)[1][-fp_features:]
                 
      
                tmp = linear.weight.data.cuda()                
                quant_linear.weight_cache.copy_(tmp[:, linear.ind].to(tmp.dtype).cuda())  



                tmp = linear.weight.data.cuda()
                tmp[:, linear.ind] = 0

                scale =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (10)).to(torch.float16).reshape((1,linear.out_features))
                quant_linear.scale_col.copy_(scale)
                tmp /= quant_linear.scale_col.T

                tmp = torch.clamp(tmp.round(), -8, 7)  
                tmp = pack_to_i4(tmp.to(torch.int8).cpu())
                quant_linear.q_weight.copy_(tmp.cuda()) 
                quant_linear.ind.copy_(linear.ind.cuda().to(torch.int32))
 


        if linear.bias is not None:
            quant_linear.bias.copy_(linear.bias.half())

        return quant_linear
    
 
         

    
    @torch.no_grad()
    def FindOutliers(self,Activation):

        
        tmp = torch.unique(torch.where((  Activation.abs() > self.sigma ))[1])
        return tmp.to(torch.int32)


    @torch.no_grad()
    def forward(self, x, cache = None, unfused = False):



        if cache is  None:
            cache = self.cache


        cache.shape = x.shape[:-1] + (self.out_features, )

        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]

        if self.weight_only is True:

            y =  w8_a16_gemm(inputs, self.q_weight, self.scale_col)

            if self.bias is not None:
                y += self.bias
            return y.reshape(cache.shape)
 
        
        if unfused:
            if self.ind.shape[0]:
                cache.activation_outliers = mixlib.ExtractOutliersAndSetToZeros(self.ind, inputs)
            cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, 
                                                        inputs.shape[0], 
                                                        self.in_features,
                                                        self.bit)  
            
        

        cache.ind = self.ind

 
        if self.add_outliers:
            if cache.x_scale[0:M].max() > self.sigma / ((  2 ** (self.bit - 1) - 1  )  )  :
                 
                ind = self.FindOutliers(inputs)
                cache.new_ind = ind
                activation_outliers = mixlib.ExtractOutliersAndSetToZeros(ind,inputs)
                if self.bit == 8:
                    weight_cache = self.q_weight[:,ind].to(torch.float16) *  self.scale_col.T
                else:
                    w = unpack_int8_to_int4(self.q_weight, ind)
                    weight_cache = w *  self.scale_col.T
 
                if self.ind.shape[0] == 0:
                    cache.activation_outliers = activation_outliers
                    self.weight_cache =  weight_cache
                else:
                    cache.activation_outliers =  torch.hstack((cache.activation_outliers,activation_outliers))
                    self.weight_cache =  torch.hstack((self.weight_cache,weight_cache))
                self.ind = torch.hstack((self.ind,ind))
                cache.ind = self.ind

                cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, M, self.in_features ,self.bit)
                
                
            self.cnt += 1
            if self.cnt >= self.cache.stop or self.ind.shape[0] > 128:
                self.add_outliers = False
             

 

        
        
        
        if self.arch == 9:
            y = mixlib.gemm(cache.q_xcache,self.q_weight,M, self.out_features, self.in_features)
            if self.ind.shape[0]:
                outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T) 
                y1 = mixlib.dequantizeInt8(y, cache.x_scale, self.scale_col, outliers_fp16, 8, M, self.out_features)
                
            else:
                y1 = mixlib.dequantizeInt8(y, cache.x_scale, self.scale_col, self.cache.zeros, 8, M, self.out_features)
                

        else:

            if self.ind.shape[0]:
                
                outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T) 
                if self.bit == 8:
                    
                    y1 = mixlib.int8FusedDequantize(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,self.in_features)  
                if self.bit == 4:
                     
                    y1 = mixlib.int4FusedDequantize(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features, 
                                                            (self.in_features ) // 2)                      
            else:
                if self.bit == 8:    
                    y1 = mixlib.int8FusedDequantize(cache.q_xcache, 
                                                    self.q_weight, 
                                                    cache.x_scale,
                                                    self.scale_col,
                                                    self.cache.zeros,
                                                    M,self.out_features,self.in_features)  

                    
                if self.bit == 4:  
                      
                    y1 = mixlib.int4FusedDequantize(cache.q_xcache, 
                                                    self.q_weight, 
                                                    cache.x_scale,
                                                    self.scale_col,
                                                    self.cache.zeros,
                                                    M,self.out_features,(self.in_features )// 2) 
        if self.bias is not None:
            y1 += self.bias
        
        #print(self.ind.shape[0])
 
        return y1.reshape(cache.shape)

    @torch.no_grad()
    def forward_without_preconditionFusedSilu(self, x, cache):
        

        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]


        if not self.forward_without_precondition_len ==  cache.ind.shape[0]:

            
            if cache.ind.shape[0]:
                ind = cache.new_ind
                if self.bit == 8:
                    weight_cache = self.q_weight[:,ind].to(torch.float16) *  self.scale_col.T
                else:
                    w = unpack_int8_to_int4(self.q_weight, ind)
                    weight_cache = w *  self.scale_col.T

                if self.ind.shape[0] == 0:
                    self.weight_cache =  weight_cache
                else:
                    self.weight_cache =  torch.hstack((self.weight_cache,weight_cache))
                self.ind = cache.ind
                self.forward_without_precondition_len = self.ind.shape[0]

 
 
      
        if self.arch == 9:
            y = mixlib.gemm(cache.q_xcache,self.q_weight,M, self.out_features, self.in_features)
            if self.ind.shape[0]:
                outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T)
                y1 = mixlib.dequantizeInt8Silu(y, cache.x_scale, self.scale_col, outliers_fp16, 8, M, self.out_features)
                
            else:
                y1 = mixlib.dequantizeInt8Silu(y, cache.x_scale, self.scale_col, self.cache.zeros, 8, M, self.out_features)
                

        else:    
            if self.bit == 8:        
                if self.ind.shape[0]:
 
                    outliers_fp16 = torch.mm( cache.activation_outliers,  self.weight_cache.T)
                
                    
                    y1 = mixlib.int8FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,self.in_features)  
                    
                else:

                    y1 = mixlib.int8FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            self.cache.zeros,
                                                            M,self.out_features,self.in_features )  

            if self.bit == 4:        
                if self.ind.shape[0]:


                    outliers_fp16 = torch.mm( cache.activation_outliers,  
                    self.weight_cache.T)
                    
                    y1 = mixlib.int4FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,
                                                            (self.in_features )// 2)  
                    
                else:
 
                    raise RuntimeError("int4 mod should have outliers !")

        if self.bias is not None:
            y1 += self.bias


        return y1.reshape(cache.shape)
    
