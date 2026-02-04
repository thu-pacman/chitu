# Stable Video Diffusion Temporal Controlnet

## Overview
Introducing the Stable Video Diffusion Temporal Controlnet! This tool uses a controlnet style encoder with the svd base. It's designed to enhance your video diffusion projects by providing precise temporal control.

## Setup
- **Controlnet Model:** you can get the depth model by running the inference script, it will automatically download the depth model to the cache, the model files can be found here: [temporal-controlnet-depth-svd-v1](https://huggingface.co/CiaraRowles/temporal-controlnet-depth-svd-v1)
- **Installation:** run `pip install -r requirements.txt`
- **Execution:** Run "run_inference.py".

## Demo

![combined_with_square_image_new_gif](https://github.com/CiaraStrawberry/sdv_controlnet/assets/13116982/055c8d3b-074e-4aeb-9ddc-70d12b5504d5)

## Notes
- **Focus on Central Object:** The system tends to extract motion features primarily from a central object and, occasionally, from the background. It's best to avoid overly complex motion or obscure objects.
- **Simplicity in Motion:** Stick to motions that svd can handle well without the controlnet. This ensures it will be able to apply the motion.

## Training
My example training config is configured like this:
```
accelerate launch train_svd.py \
 --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
 --output_dir="model_out" \
 --csv_path="path-to-your-csv" \
 --video_folder="path-to-your-videos" \
 --depth_folder="path-to-your-depth" \
 --motion_folder="path-to-your-motion" \
 --validation_image_folder="./validation_demo/rgb" \
 --validation_control_folder="./validation_demo/depth" \
 --width=512 \
 --height=512 \
 --learning_rate=2e-5 \
 --per_gpu_batch_size=8 \
 --num_train_epochs=5 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=2 \
 --checkpointing_steps=2000 \
 --validation_steps=400 \
 --gradient_checkpointing
```

## Acknowledgements
- **lllyasviel:** for the original controlnet implimentation
- **Stability:** for stable video diffusion.
- **Diffusers Team:** For the svd implementation.
- **Pixeli99:** For providing a practical svd training script: [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend)
