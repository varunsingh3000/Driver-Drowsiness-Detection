#  Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection

This is the repository for the implemenation of the Driver Drowsiness Detection problem presented in the paper "[A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection](https://arxiv.org/abs/1904.07312)". The model used was Hierarchical Multiscale Long Short-Term Memory (HM-LSTM) and details about it can be found in the above mentioned paper. This repository is created for the CS4245 Seminar Computer Vision by Deep Learning (2022/23 Q4) and provides the instructions for the setup of the repository and describes our implementation in detail. The blog describing our approach and methodology can be found [here](https://hackmd.io/s7w_NxOMSiCWSou_JPYbAw). The original details and description of the paper repository can be found [here](https://github.com/rezaghoddoosian/Early-Drowsiness-Detection).


### These codes were tested on Ubuntu 22.04 with tensorflow version 1.15.0

The supporting code and data used for the paper:"A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection":

This proposed temporal model uses blink features to detect both early and deep drowsiness with an intermediate regression step, where drowsiness is estimated with a score from 0 to 10. 

## Instruction to Run the Code:
*THESE CODES WERE APPLIED ON THE UTA-RLDD DATASET*

*You can refer to the comments inside each .py file for more detailed information*

### Make sure all .py files are downloaded then install all the required packages. 

tensorflow==1.15
protobuf==3.20.*
numpy==1.19.5

For reference these are all the packages we had in our virtual enevironment but the important ones are the three mentioned above.

absl-py              1.4.0
astor                0.8.1
gast                 0.2.2
google-pasta         0.2.0
grpcio               1.54.2
h5py                 3.8.0
importlib-metadata   6.6.0
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.2
Markdown             3.4.3
MarkupSafe           2.1.2
numpy                1.19.5
opt-einsum           3.3.0
pip                  22.0.2
protobuf             3.20.3
setuptools           59.6.0
six                  1.16.0
tensorboard          1.15.0
tensorflow           1.15.0
tensorflow-estimator 1.15.1
termcolor            2.3.0
typing_extensions    4.5.0
Werkzeug             2.2.3
wheel                0.37.1
wrapt                1.15.0
zipp                 3.15.0

#### Citation:
All documents (such as publications, presentations, posters, etc.) that report results, analysis, research, or equivalent that were obtained by using this source should cite the following research paper: https://arxiv.org/abs/1904.07312

    @inproceedings{ghoddoosian2019realistic,
    title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
    author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
    pages={0--0},
    year={2019}
    }


#### Link to the UTA-RLDD dataset:

https://sites.google.com/view/utarldd/home

#### Link to the demo on youtube:
https://youtu.be/3psnER2oVUA

"I hope the result of this work and the UTA-RLDD dataset can pave the way for a more unified research on drowsiness detection , so the result of one work could be compared to the others based on a realistic drowsiness dataset.": R.G
