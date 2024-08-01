class MixLinear_GEMM:
    def __init__(self, in_features, out_features):
        super().__init__()
        
 
        self.in_features = in_features
        self.out_features = out_features


        
    def __str__(self):
        return  "MixLinear_GEMM(in_features = %d, out_features = %d, weight_only = %d)"%(self.in_features,self.out_features, int(self.weight_only)) 
    

a = MixLinear_GEMM(2,3)
print(a)