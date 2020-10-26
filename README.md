# DSTA-Net
Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition in ACCV2020

# Note


# Data Preparation

 - SHREC/DHG
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
    
To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer. 

    `python train_val_test/eval.py --config ./config/val/shrec_dstanet_14.yaml`

Then combine the generated scores with: 

    `python train_val_test/train.py --datasets ntu/xview`
     
# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{2sagcn2019cvpr,  
          title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},  
          author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},  
          booktitle = {CVPR},  
          year      = {2019},  
    }
    
    @article{shi_skeleton-based_2019,
        title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
        journal = {arXiv:1912.06971 [cs]},
        author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
        month = dec,
        year = {2019},
	}
# Contact
For any questions, feel free to contact: `lei.shi@nlpr.ia.ac.cn`