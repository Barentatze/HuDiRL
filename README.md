# HuDiRL: Human-Guided Diffusion via Reinforcement Learning for Single-Image Generation


## Setting up Environment
To setup the enviroment, execute the following command:
```bash
source setup.sh
```

## Training the model
To train the model, execute the training scripts in the scripts folder or the following command. 
```bash
export LOGDIR="OUTPUT/sinddpm-yourimage-day-commitseq"

mpiexec -n 8 python image_train.py --data_dir data/yourimage.png --lr 5e-4 --diffusion_steps 1000 --image_size 256
                                   --noise_schedule linear --num_channels 64 --num_head_channels 16 --channel_mult "1,2,4" 
                                   --attention_resolutions "2" --num_res_blocks 1 --resblock_updown False --use_fp16 True 
                                   --use_scale_shift_norm True --use_checkpoint True --batch_size 16
```
The experimental results are then saved in the folder ./OUTPUT/sinddpm-(yourimage)-(day)-(commitseq).
Training on 1 NVIDIA Tesla H100 is recommended. 

## Testing the model
To test a trained model, execute the testing scripts in the scripts folder or the following command. 
```bash
python image_sample.py --data_dir data/yourimage.png --diffusion_steps 1000 --image_size 256 --noise_schedule linear
                       --num_channels 64 --num_head_channels 16 --num_res_blocks 1 --channel_mult "1,2,4"
                       --attention_resolution "2" --resblock_updown False --use_fp16 True --use_scale_shift_norm True 
                       --use_checkpoint True --model_root OUTPUT/sinddpm-yourimage-day-commitseq 
                       --results_path RESULT/sinddpm-yourimage-day-commitseq/
```
These testing results are then saved in the folder ./RESULT/sinddpm-(yourimage)-(day)-(commitseq)/.

## Acknowledge
Our code is developed based on [guided-diffusion](https://github.com/openai/guided-diffusion) and [SinDiffusion](https://github.com/WeilunWang/SinDiffusion)
