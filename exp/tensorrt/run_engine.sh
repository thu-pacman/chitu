for bs in 2 4 8; do 
    echo "Batch size = $bs"
    trtexec --loadEngine=unet.${bs}.fp16.engine --shapes=sample:${bs}x4x64x64,timestep:${bs},encoder_hidden_states:${bs}x64x768 --fp16 --separateProfileRun
done 
