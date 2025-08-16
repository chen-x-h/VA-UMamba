## Implementation of VA-UMamba


### 
## VA-UMamba pytorch implementation

model base on https://github.com/bowang-lab/U-Mamba

for training: https://github.com/MIC-DKFZ/nnUNet

#
## Prepare Data
See in preprocess/*  

Use 3dircadb as a sample  

The 3D-IRCADb dataset can be downloaded from the official website:   
https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/  

1. preprocess/pre_process.py for process  
2. preprocess/data_prepare.py for generating a nnUNet-style dataset  

#
## Integrate into nnU-Net
See in model/* and loss/*  

You can get a model from model/VAUMamba.py.   
Put model in nnUnet in your way: nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py build_network_architecture  
Notice: you should change the nnUNetPlans.json refer to plan/* and set os.environ['nnUNet_compile'] = 'False' 

DDLoss in loss/deep_supervision.py  
You can put it in nnunetv2/training/loss/deep_supervision.py  

#
## Model Training and Inference

Refer to nnUnet: https://github.com/MIC-DKFZ/nnUNet

#
## Metrics

See in metrics/val_metrics.py

