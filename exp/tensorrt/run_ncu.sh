for bs in 1 2 4; do
    ncu -f -o unet_model_bs${bs}_bf16 --replay-mode application --nvtx --profile-from-start off python3 ./run_trt_engine.py $bs
done
