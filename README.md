# Recurrent Transformer

## Ver 1.0（PyTorch）

#### Project created by Jiewen Yang

This project is created by Jiewen Yang. Our experimental platform is configured with 6 RTX3090 (cuda11.2), 256G RAM and Intel (R) Xeon (R) gold 6226R. In this demo code, we use a two-layer RViT trained in Jester dataset as example.



## Install 

You need to build the relevant environment first, please refer to : [**requirements.txt**](requirements.txt)

It is recommended to use Anaconda to establish an independent virtual environment, and python > = 3.6.3; (3.8.0 is used for this experimental platform).



For the installation of APEX training framework provided by NVIDIA, please refer to : **https://github.com/NVIDIA/apex**



## Data Preparation

This project provides the use case of video classification task;

The address index of the dataset can be found in the **main.py**, where you could do the parameters modification;

For different tasks, the composition of data sets have significant different, so there is no repetition in this file;



## 1. Download The *Jester* Dataset

In order to train the gesture recognition system, we will use TwentyBN's [Jester Dataset](https://www.twentybn.com/datasets/jester). This dataset consists of 148,092 labeled videos, depicting 27 different classes of human hand gestures. This dataset is made available under the Creative Commons Attribution 4.0 International license CC BY-NC-ND 4.0. It can be used for academic research free of charge. In order to get access to the dataset you will need to register.

The Jester dataset is provided as one large TGZ archive and has a total download size of 22.8 GB, split into 23 parts of about 1 GB each. After downloading all the parts, you can extract the videos using:

```
cat 20bn-jester-v1-?? | tar zx
```

The CSV files containing the labels for the videos in the Jester dataset have already been downloaded for you and can be found in the **20bn-jester-v1/annotations** folder.

In the **20bn-jester-v1/annotations** folder you will find the following CSV files:

- `jester-v1-labels-quick-testing.csv`
- `jester-v1-train-quick-testing.csv`
- `jester-v1-validation-quick-testing.csv`

More information, including alternative ways to download the dataset, is available in the [Jester Dataset](https://www.twentybn.com/datasets/jester) website.



## Frame Code Composition

The framework consists of the following simplified folders and files：

```shell
│  main.py
|  visualize_demo.py
│
├─dataset
       data_loader.py
       data_parser.py
       data_utils.py
       ...
       transform.py
       transforms_video.py
|       
├─models
       yourmodel.py
│
├─result
   └─writer
   └─attention_visual
|
└─tools
        init.py
        utils.py
        visualizer.py
        ...
        xxx.py
```

> main.py -> which enable the training and validation, also the main program；
>
> models ->  where you store your models code；
>
> result   -> here stores your training results or other generated files;
>
> tools     -> the function set needed for training;



## Training（Currently Do Not Provide）

In this framework, after the parameters are configured in the file **args.py**, you only need to use the command:

```shell
python -m torch.distributed.launch --nproc_per_node=n train.py
```

Then you can start distributed training, where **n** is the number of processes you need, and each process will be assigned a graphics card independently;

**Note: ** Please set the number of graphics cards you need and their ID in parameter "--enable_GPUs_id" .

For example, we train the model on Jester V1 dataset:

```shell
python -m torch.distributed.launch --nproc_per_node=2 train.py --train --enable_GPUs_id=[0,1] --data_dir=./path_to_dataset --label_dir=./patch_to_label 
```

If you only need to use single card or multi card parallel training, just set -- distributed to False in the configuration;

Then use the command:

```shell
python train.py --train enable_GPUs_id=[your_target_gpu] --data_dir=./path_to_dataset --label_dir=./patch_to_label --model_path=./path_to_the_trained_model
```



## Evaluation

Make sure you [install](https://pytorch.org/get-started/previous-versions) the stable version of pytorch (we use 1.10.0 + cu113).

```shell
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Just simply use this command:

```shell
python main.py --model_path=./model/final.pth
```

The network can be printed as follow:

```shell
RecursiveScriptModule(
  original_name=RViT
  (embedding): RecursiveScriptModule(original_name=Conv2d)
  (transformer): RecursiveScriptModule(
    original_name=Encoder
    (dropout): RecursiveScriptModule(original_name=Dropout)
    (layers): RecursiveScriptModule(
      original_name=ModuleList
      (0): RecursiveScriptModule(
        original_name=ModuleList
        (0): RecursiveScriptModule(
          original_name=Encoder1DBlock
          (layer_norm_input): RecursiveScriptModule(original_name=LayerNorm)
          (layer_norm_hidden): RecursiveScriptModule(original_name=LayerNorm)
          (layer_norm_out): RecursiveScriptModule(original_name=LayerNorm)
          (attention): RecursiveScriptModule(
            original_name=MultiHeadDotProductAttention
            (to_qkv): RecursiveScriptModule(original_name=Linear)
            (to_qkv_h): RecursiveScriptModule(original_name=Linear)
            (elu): RecursiveScriptModule(original_name=ELU)
            (to_out): RecursiveScriptModule(
              original_name=Sequential
              (0): RecursiveScriptModule(original_name=Linear)
              (1): RecursiveScriptModule(original_name=Dropout)
            )
          )
          (mlp): RecursiveScriptModule(
            original_name=FeedForward
            (net): RecursiveScriptModule(
              original_name=Sequential
              (0): RecursiveScriptModule(original_name=Linear)
              (1): RecursiveScriptModule(original_name=GELU)
              (2): RecursiveScriptModule(original_name=Dropout)
              (3): RecursiveScriptModule(original_name=Linear)
              (4): RecursiveScriptModule(original_name=Dropout)
            )
          )
          (drop_out_attention): RecursiveScriptModule(original_name=Dropout)
        )
      )
      (1): RecursiveScriptModule(
        original_name=ModuleList
        (0): RecursiveScriptModule(
          original_name=Encoder1DBlock
          (layer_norm_input): RecursiveScriptModule(original_name=LayerNorm)
          (layer_norm_hidden): RecursiveScriptModule(original_name=LayerNorm)
          (layer_norm_out): RecursiveScriptModule(original_name=LayerNorm)
          (attention): RecursiveScriptModule(
            original_name=MultiHeadDotProductAttention
            (to_qkv): RecursiveScriptModule(original_name=Linear)
            (to_qkv_h): RecursiveScriptModule(original_name=Linear)
            (elu): RecursiveScriptModule(original_name=ELU)
            (to_out): RecursiveScriptModule(
              original_name=Sequential
              (0): RecursiveScriptModule(original_name=Linear)
              (1): RecursiveScriptModule(original_name=Dropout)
            )
          )
          (mlp): RecursiveScriptModule(
            original_name=FeedForward
            (net): RecursiveScriptModule(
              original_name=Sequential
              (0): RecursiveScriptModule(original_name=Linear)
              (1): RecursiveScriptModule(original_name=GELU)
              (2): RecursiveScriptModule(original_name=Dropout)
              (3): RecursiveScriptModule(original_name=Linear)
              (4): RecursiveScriptModule(original_name=Dropout)
            )
          )
          (drop_out_attention): RecursiveScriptModule(original_name=Dropout)
        )
      )
    )
  )
  (to_cls_token): RecursiveScriptModule(original_name=Identity)
  (classifier): RecursiveScriptModule(
    original_name=Sequential
    (0): RecursiveScriptModule(original_name=LayerNorm)
    (1): RecursiveScriptModule(original_name=Linear)
  )
)

```



## Visualize（Currently Do Not Provide）

For the attention visualization, we take https://github.com/luo3300612/Visualizer as our reference and implement this code to our project.

```shell
git clone https://github.com/luo3300612/Visualizer
cd Visualizer
python setup.py install
```

Set the --visualize to True will enable the Visualization mode, the attention visualize figure would be saved at ./result/attention_visual:

```shell
python main.py --visualize --model_path=./result/model.pth
```



## Semi-Precision and Full Precision Training

The framework provides a variety of training modes;

If you need to settings the modes, please change the parameter **'-- opt_ level'** in file **args.py** ;

| Mode | Training Precision Type     |
| ---- | --------------------------- |
| O0   | Full Precision（FP32）      |
| O1   | Semi-Precision（FP32+FP16） |
| O2   | Half-Precision（FP16）      |

In most cases, we recommend to use Semi-precision training mode, which is O1;



Note: In this demo we do not provide the apex code for training or inference.
