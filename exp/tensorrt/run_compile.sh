# 512x512
# for bs in 4; do 
#     echo "Batch size = $bs"
#     trtexec --onnx=/home/zly/Works/ModelZoo/onnx/diffusion/models/unet/1/unet.onnx --saveEngine=unet.${bs}.fp16.512x512.engine --shapes=sample:${bs}x4x64x64,timestep:${bs},encoder_hidden_states:${bs}x64x768 --fp16 --dumpProfile --dumpLayerInfo --exportTimes=times.$bs.log --exportProfile=profile.$bs.log --exportLayerInfo=layerinfo.$bs.log --separateProfileRun
# done 

# 256x256
for bs in 8 16; do 
    echo "Batch size = $bs"
    trtexec --onnx=/home/zly/Works/ModelZoo/onnx/diffusion/models/unet/1/unet_dynamic_shape.onnx --saveEngine=unet.${bs}.fp16.256x256.engine --shapes=sample:${bs}x4x32x32,timestep:${bs},encoder_hidden_states:${bs}x64x768 --fp16 --dumpProfile --dumpLayerInfo --exportTimes=times.$bs.log --exportProfile=profile.$bs.log --exportLayerInfo=layerinfo.$bs.log --separateProfileRun
done 
