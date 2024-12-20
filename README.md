# Moxin-XD


## Model Download

You can download our efficient stable diffusion model from this [link](https://huggingface.co/piuzha/efficient_sd). It is located on Huggingface with 'piuzha/efficient_sd'.

We adopt a more efficient SD model. Our model architecture is based on the SDv1.5 architecture. A few blocks in the original SD1.5 model architecture are removed. 


## Requirements

#### Diffusers package
Follow the [diffusers](https://huggingface.co/docs/diffusers/en/installation) package to install the environment.

#### Mac configuration

For Mac users, you can still follow the [diffusers](https://huggingface.co/docs/diffusers/en/installation) package to install the environment. As long as you can run the original SD v1.5 model successfully, you can seamlessly replace the original SD v1.5 model with our model. Specifically, to install the environment, you can  follow the instructions below,
```
conda install -n sd python=3.10 -y
conda activate sd
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install diffusers["torch"] transformers
pip install accelerate
pip install git+https://github.com/huggingface/diffusers
```



## Inference

Run the following command to run inference with the model. Specify the model directory in the file
```
$ python inference.py
```

Specifically, you can load the model through 
```
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("piuzha/efficient_sd/", use_safetensors=True).to("cuda")
```
Then run the model to generate images such as
```
image = pipeline("An astronaut riding a horse, detailed, 8k", num_inference_steps=25).images[0]
image.save('test.png')
```


## Training Datasets

To prepare the dataset, you can install the img2dataset package.
```
pip install img2dataset
```

There are multiple datasets available. The scripts to download the datasets are located under the dataset_examples directory. You can refer to the specific script for details. 



## Training Script

We follow  a stand  method to train the stable diffusion model. You can refer to the [huggingface diffusers text_to_image](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) script to train the text2image diffusion model. 

For example, you can finetune the model with the following command,
```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model" \
  --push_to_hub
```

More details about the training of the diffusion model can be find [here](https://huggingface.co/docs/diffusers/en/training/text2image).


## SD Webui

Our model can be used in [SD Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). 

#### 1. SD Webui install

Follow this [link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/master) to install the webui environment. 

Specifically, you can follow the follwoing instructions.
```
sudo apt install git software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10-venv -y
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui && cd stable-diffusion-webui
python3.10 -m venv venv
./webui.sh
```

#### 2. Prepare model

You need to download the model from this [link](https://huggingface.co/piuzha/efficient_sd). Put the model under the 'stable-diffusion-webui/models/Stable-diffusion/' directory.  The model is obtained by converting our diffusers model to the compvis model through this file 'scripts/convert_diffusers_to_original_stable_diffusion.py' which can be obtained from this [link](scripts/convert_diffusers_to_original_stable_diffusion.py). 

#### 3. Prepare config file

You also need to use an  updated config file for our model to replace the original config file 'stable-diffusion-webui/configs/v1-inference.yaml'.  The new config file can be found under our 'configs/v1-inference.yaml'. 


## Performance Comparison

We compare our model with the original SD v1.5 model. The batch size is set to 1 for all methods.  Our model can achieve faster inference. 

|     model   |   Inference Time  |   Steps   |  Device      |
|-------------|-------------------|-----------|--------------|
| Ours        |   4.5s            |   20      |  1080Ti      |
|Ours         |    2.8s           |   20      | Titan RTX    |
| SD v1.5     |   6.7s            | 20        |  1080Ti      |
| SD v1.5     |   4.6s            | 20        |  Titan RTX   |


