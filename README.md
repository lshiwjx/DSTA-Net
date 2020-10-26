# DSTA-Net
Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition in ACCV2020

# Result
A little different with paper due the reimplementation.

 - NTU-60-CS: ~91.8%
 - SHREC-14: ~97.2%

# Data Preparation

 - SHREC
    - Download the SHREC data from http://www-rech.telecom-lille.fr/shrec2017-hand/
    - Generate the train/test splits with `python prepare/shrec/gendata.py`
 - DHG
    - Download the DHG data from the http://www-rech.telecom-lille.fr/DHGdataset/
    - Generate the train/test splits with `python prepare/dhg/gendata.py`
 - NTU-60
    - Download the NTU-60 data from the https://github.com/shahroudy/NTURGB-D
    - Generate the train/test splits with `python prepare/ntu_60/gendata.py`
 - NTU-120
    - Download the NTU-120 data from the https://github.com/shahroudy/NTURGB-D
    - Generate the train/test splits with `python prepare/ntu_120/gendata.py`
 - Note
    - You can check the raw/generated skeletons through the function `view_raw/generated_skeletons_and_images()` for NTU and function `ske_vis()` for dhg/shrec in gendata.py
     
# Training & Testing

Change the config file depending on what you want.

    `python train_val_test/train.py --config ./config/shrec/shrec_dstanet_14.yaml`

Train with decoupled modalities by changing the 'num_skip_frame'(None to 1 or 2) option and 'decouple_spatial'(False to True) option in config file and train again. 
    
Then combine the generated scores with: 

    `python train_val_test/ensemble.py`
     
# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{dstanet_accv2020,  
          title     = {Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition},  
          author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
          booktitle = {ACCV},  
          year      = {2020},  
    }
    
# Contact
For any questions, feel free to contact: `lei.shi@nlpr.ia.ac.cn`