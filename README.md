# DGarment
This repository contains a tensorflow implementation of "High-Quality Animatable Dynamic Garment Reconstruction from Monocular Videos.

[Project Page](http://cic.tju.edu.cn/faculty/likun/projects/DGarment/index.html)



<!-- # Requirement -->



# Installation
Create virtual environment

    conda create -n dgarment python=3.7
    conda activate dgarment

Install cuda and cudnn

    conda install cudatoolkit=10.1

Install tensorflow

    pip install tensorflow-gpu==2.1

Install protobuf

    pip install protobuf==3.20

Install dirt:  [https://github.com/pmh47/dirt](https://github.com/pmh47/dirt)

Install other environments

    pip install -r requirements.txt 

Download the neutral SMPL model from [http://smplify.is.tue.mpg.de/](http://smplify.is.tue.mpg.de) and place it in the "Data/smpl" folder (Data/smpl/model_f.pkl, Data/smpl/model_neutral.pkl, Data/smpl/model_m.pkl).
    

Download Pre-processed data sample from [here](https://drive.google.com/drive/folders/1suTuBf8TKgCrxJxmvdDV6JW2VSHswnBi?usp=sharing) and place them in the "person" folder.
    


# Train
    python train.py


# Test
    python eval.py

# Data preparation
If you want to process your own data, some pre-processing steps ([Cloth-Segmentation](https://github.com/levindabhi/cloth-segmentation), [BCNet](https://github.com/jby1993/BCNet), [PyMAF](https://github.com/HongwenZhang/PyMAF)) are needed:

    1.Remove the background of the video and crop it to 512*512.
    2.Run PyMAF to estimate SMPL parameters and run /person/makenpy.py to make train data.
    3.Run Cloth-Segmentation on the image to get cloth mask.
    4.Run BCNet on the first frame to get body.mat and garment.obj
    5.Run /person/pifuhd.py to get normal image.


# Citation
Please cite the following paper if it helps your research:

    @article{li2023tcsvt,
      author = {Xiongzheng Li and Jinsong Zhang and Yu-Kun Lai and Jingyu Yang and Kun Li},
      title = {High-Quality Animatable Dynamic Garment Reconstruction from Monocular Videos},
      journal = {IEEE Transactions on Circuits and Systems for Video Technology},
      year={2023},
    }


# Contact
For more questions, please contact lxz@tju.edu.cn

# License
        Software Copyright License for non-commercial scientific research purposes
        Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this License

        License Grant
        Licensor grants you (Licensee) personally a single-user, non-exclusive, non-transferable, free of charge right:

        To install the Model & Software on computers owned, leased or otherwise controlled by you and/or your organization;
        To use the Model & Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects;
        Any other use, in particular any use for commercial, pornographic, military, or surveillance, purposes is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artifacts for commercial purposes. The Data & Software may not be used to create fake, libelous, misleading, or defamatory content of any kind excluding analyses in peer-reviewed scientific research. The Data & Software may not be reproduced, modified and/or made available in any form to any third party.

        The Data & Software may not be used for pornographic purposes or to generate pornographic material whether commercial or not. This license also prohibits the use of the Software to train methods/algorithms/neural networks/etc. for commercial, pornographic, military, surveillance, or defamatory use of any kind. By downloading the Data & Software, you agree not to reverse engineer it.

        No Distribution
        The Model & Software and the license herein granted shall not be copied, shared, distributed, re-sold, offered for re-sale, transferred or sub-licensed in whole or in part except that you may make one copy for archive purposes only.

        Disclaimer of Representations and Warranties
        You expressly acknowledge and agree that the Model & Software results from basic research, may contain errors, and that any use of the Model & Software is at your sole risk. 
        
# Acknowledgments
The codes of DGarment are largely borrowed from [PBNS](https://github.com/hbertiche/PBNS).

