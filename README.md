#  Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection

This is the repository for the implemenation of the Driver Drowsiness Detection problem presented in the paper "[A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection](https://arxiv.org/abs/1904.07312)". The model used was Hierarchical Multiscale Long Short-Term Memory (HM-LSTM) and details about it can be found in the above mentioned paper. This repository is created for the CS4245 Seminar Computer Vision by Deep Learning (2022/23 Q4) and provides the instructions for the setup of the repository and describes our implementation in detail. The blog describing our approach and methodology can be found [here](https://hackmd.io/s7w_NxOMSiCWSou_JPYbAw). The original details and description of the paper repository can be found [here](https://github.com/rezaghoddoosian/Early-Drowsiness-Detection).


### These codes were tested on Ubuntu 22.04 with tensorflow version 1.15.0

The supporting code and data used for the paper:"A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection":

This proposed temporal model uses blink features to detect both early and deep drowsiness with an intermediate regression step, where drowsiness is estimated with a score from 0 to 10. 

## Instruction to Run the Code:
*THESE CODES WERE APPLIED ON THE UTA-RLDD DATASET*

*You can refer to the comments inside each .py file for more detailed information*

### Make sure all .py files are downloaded then install all the required packages. 

1. tensorflow==1.15
2. protobuf==3.20.*
3. numpy==1.19.5

For reference these are the important packages needed to run the repository, if any other packages are missing, install them and there shouldn't be any compatability issues.

## Introduction

Drowsiness detection is crucial in domains such as driving and workplace safety, as it helps mitigate the risks associated with fatigue-related accidents and fatalities. The goal of the original paper [1] is to create practical and easily deployable systems that can identify drowsiness at an early stage. 

In this blog post, we will delve into the key aspects of the original paper, providing a comprehensive overview of its content. The paper introduces a new and publicly available dataset, known as the Real-Life Drowsiness Dataset (RLDD). This dataset addresses the limitations of existing datasets by incorporating real-life scenarios, diverse subjects, and a substantial amount of video data. It includes videos capturing both subtle and obvious signs of drowsiness, making it highly valuable for studying early drowsiness detection.

The paper also presents a baseline method called the Hierarchical Multiscale Long Short-Term Memory (HM-LSTM) network, which utilizes blink features to detect drowsiness. This computational model leverages the temporal relationship between blinks to identify patterns indicative of drowsiness. The experimental results indicate that the proposed method outperforms human judgment in terms of accuracy for drowsiness detection, demonstrating its effectiveness in capturing early signs of drowsiness that are often missed by human observers.

Throughout this blog post, we will explore the paper's contributions, discuss related work in the field of drowsiness detection, and provide an overview of the RLDD dataset and the proposed baseline method. We will also delve into the experiments conducted to evaluate the model's performance and discuss the results obtained. Additionally, we will highlight the conclusions that we arrived at and challenges we faced during implementation.



## Related Work

In this section, we delve into the research conducted on drowsiness detection and elucidate the challenges encountered by previous studies. [2]] focuses on driver drowsiness detection using Convolutional Neural Networks (CNNs). It demonstrates that models trained on publicly available datasets can suffer from overfitting and exhibit racial bias. The authors propose a novel visualization technique using Principal Component Analysis (PCA) to identify groups of people where there might be potential discrimination and combine it with a model accuracy overlay.
 
Consensus among researchers [6,7] supports the existence of three primary sources of information for drowsiness detection: performance measurements, physiological measurements, and behavioral measurements.[3] aims to investigate the relationship between eye blink metrics, vigilance performance, and cerebral blood flow velocities. The results suggest that eye blink information can serve as an indicator of arousal levels, and using an eye-tracker in operational environments could enable the implementation of preventative measures or cognitive augmentation techniques based on detecting changes in eye blinks. [9] highlights the limited research on analyzing the temporal dynamics of facial expressions for drowsiness detection. The authors propose a new method utilizing Hidden Markov Models (HMMs) to analyze facial expressions and detect drowsiness, with experimental results validating the effectiveness of their approach. [10] addresses the limitations of existing vision-based drowsy driver alert systems and the lack of public datasets for evaluating different methods. The authors propose a hierarchical temporal Deep Belief Network (HTDBN) approach to extracts facial and head features and present a comprehensive dataset with diverse driver videos to validate the effectiveness of their HTDBN framework across various visual cues.
 
State-of-the-art benchmarks do not use temporal features, and these are incorporated in [1]. Although physiological measurements like heart rate, electrocardiogram (ECG), electromyogram (EMG), electroencephalogram (EEG)[4,5], and electrooculogram (EOG)[4] have been utilized for drowsiness monitoring, their intrusiveness and impracticality in car or workspace settings limit their usage, despite their high accuracy. The DROZY dataset [8] includes various drowsiness-related signals like EEG, EOG, and NIR images, obtained from genuinely drowsy subjects. However, the RLDD dataset [1] offers three advantages over DROZY: a larger number of subjects, data capturing each subject in all three alertness classes, and diverse recording conditions using personal cell phones with different backgrounds and color video.

## Data

![data](https://github.com/varunsingh3000/Driver-Drowsiness-Detection/assets/64498789/e1d1186c-f950-493a-aa42-23f88450f1fa)

The RLDD dataset [1] used in this project focuses on drowsiness detection and includes 60 healthy participants. The participants were instructed to record three videos of themselves in different drowsiness states based on the Karolinska Sleepiness Scale (KSS). The dataset contains 180 RGB videos, approximately ten minutes each, categorized into three classes: "alert" (0), "low vigilant" (5), and "drowsy" (10). Participants provided the labels based on their subjective assessment while recording the videos. The dataset's total size is 111.3 gigabytes, and it was recorded using personal cell phones or web cameras, resulting in variations in video resolutions and qualities. Cross-validation was performed with five folds of 12 participants each.

## Baseline Model

A multi-stage pipeline was used for drowsiness detection. The stages were:
1. Blink Detection and Blink Feature Extraction: use of blink-related features like duration, amplitude, and eye opening velocity to capture temporal patterns in human eyes. It uses a pre-trained face detector to detect blinks and extracts blink features. A post-processing step is applied to identify multiple blinks in a single detection.

![model](https://github.com/varunsingh3000/Driver-Drowsiness-Detection/assets/64498789/2d7675f1-1fda-4823-bb22-7ce2e5e23751)

2. Drowsiness Detection Pipeline:
        a. Preprocessing: To handle individual differences in blinking patterns, the features are normalized across subjects. The first third of blinks in an alert state video is used to compute mean and standard deviation for normalization.
        b. Feature Transformation Layer: The blink features are transformed into a higher-dimensional space using a fully connected layer, allowing the network to learn relevant representations.
        c. HM-LSTM Network: A hierarchical multiscale LSTM network is used to capture the temporal patterns in blinking. This network considers the relationship between consecutive blinks and discovers the underlying hierarchical structure in a blink sequence.
        d. Fully Connected Layers: Additional fully connected layers capture the results of the HM-LSTM network from different perspectives.
        e. Regression Unit: A single node at the end of the network outputs a real number from 0 to 10, indicating the degree of drowsiness based on the input blinks.
        f. Discretization and Voting: The regression output is discretized into predefined classes (e.g., Alert, LowVigilant, Drowsy). The most frequent predicted class from multiple blink sequences determines the final classification result of a video.
        g. Loss Function: The model minimizes a loss function that penalizes inaccurate predictions quadratically, emphasizing correct classification rather than perfect regression.
    
## Approach


We aimed to explore and compare different variations of the above described baseline model to improve the drowsiness detection system. These variations involve testing different architectures and configurations. Specifically, we experimented with the following model variations:

1. HMLSTM with different hyperparameters: The baseline model utilizes a hierarchical multiscale LSTM (HM-LSTM) network to capture temporal patterns in blinking. To further optimize the model's performance, we tested different hyperparameters. By adjusting hyperparameters such as the number of LSTM layers, hidden units, learning rate, or regularization techniques,we assessed the impact of these changes on the model's ability to detect drowsiness accurately.
2. Vanilla LSTM: In this variation, we removed the hierarchical aspect of the HM-LSTM network and use a standard LSTM network. This modification simplified the architecture by eliminating the consideration of hierarchical structure in the blink sequence. By comparing the performance of the vanilla LSTM with the HM-LSTM,we evaluated the necessity and effectiveness of the hierarchical approach for drowsiness detection.
3. Fully connected (FC) layers only: In this variation, the LSTM component was entirely removed from the model, and only FC layers were utilized. FC layers allowed for more direct interactions between the input features and the output classification. In this case, we assessed whether the temporal patterns captured by the LSTM network contribute significantly to the drowsiness detection performance or if the FC layers alone could achieve comparable results.

By exploring these different model variations, we gained some insights into the strengths and weaknesses of each approach. It helped us identify the most effective model configuration for drowsiness detection.

### Evaluation Metrics
The following four four evaluation metrics were used to assess the model's performance from different perspectives at different stages of the pipeline.

1. Blink Sequence Accuracy (BSA): This metric measures the accuracy of the model's results before the "voting stage" and after "discretization" across all test blink sequences. It evaluates how well the model classifies individual blink segments.
2. Blink Sequence Regression Error (BSRE): BSRE quantifies the regression error of the model by penalizing wrongly classified blink sequences. It considers the distance of the regressed output to the nearest true state border ( $S_i$). This metric aims to measure the accuracy of the model's blink sequence regression.
    $$BSRE = \frac{\sum_{i=1}^M C^s_i| out_i - S_i | ^2 }{M} $$
where $C^s_i$ is a binary value, equal to 0 if the i-th blink segment has been classified correctly, and 1 otherwise.
3. Video Accuracy (VA): VA is the main metric used to assess overall accuracy. It measures the percentage of entire videos (not individual segments) that are correctly classified by the model.
4. Video Regression Error (VRE): VRE calculates the regression error for videos. It penalizes wrongly classified videos based on the distance between the regressed output and the true state border. This metric indicates the margin of error for misclassified videos.



## Results and Discussion

In this section, we evaluate the baseline model described in the original paper, compare the values and perform hyperparameter tuning. Table 1 describes the results obtained in the original paper. 

##### Table 1: Original Paper results
|Model number| Model  | BSRE | VRE | BSA | VA  |
| --------| -------- | -------- | -------- |-------- | -------- |
| 1 | HM-LSTM network   | 1.90 | 1.14 | 54% | 65.2% |
| 2 | LSTM network   | 3.42 | 2.68 | 52.8% | 61.4% |
| 3 | Fully connected layers  | 2.85 | 2.17 | 52%  | 57%|

We trained the model using the features that were extracted and provided by the authors. Table 2 encapsulates the results that were obtained.

##### Table 2: Reproduction results
| Model number | Model Description |  Hyperparameters | BSRE | VRE | BSA | VA  | 
| -------- | --------------- | ---- | --- | --- | --- | -------- |
| 1        | HM-LSTM Network with default parameters (as described in the original paper)  |   LR = 0.000053, $\Delta$=1.253, batch_size=64, num_epochs=80, L2 reg coeff = 0.1 | 1.63     |  0.78   | 55.0%    | 61.1%    | 
| 2        |  HM-LSTM with tuned LR        |LR = 0.01 | 2.65     |  1.13   | 44.70%    | 57.57%    |  
| 3        |  HM-LSTM with tuned $\Delta$   |$\Delta$ = 1.0 |   0.96   |  0.27   | 55.50%    |  48.48%   |  
| 4        | FC layers with default parameters (as described in the original paper)       |LR = 0.000053, $\Delta$=1.253, batch_size=64, num_epochs=80, L2 reg coeff = 0.1 |   1.34   | 0.50    | 48.25%    |  54.72%   |  
| 5        | FC layers with tuned LR |LR = 0.01 |  2.11    |  0.94   |  52.97%   | 48.92%    |  
|  6        |FC layers with tuned $\Delta$  | $\Delta$ = 1.0 | 1.30     |  0.92   | 54.16%     |  41.56%    | 
| 7        | LSTM with default parameters (as described in the original paper)   | LR = 0.000053, $\Delta$=1.253, batch_size=64, num_epochs=80, L2 reg coeff = 0.1 |   2.97   | 0.50    | 51.15%    |  58.23%   |  
| 8        | LSTM with tuned LR      | LR = 0.01 |  2.83   |  1.45   |  47.67%   | 53.92%    | 
|  9        | LSTM with tuned $\Delta$       |  $\Delta$ = 1.0 | 1.89     |  1.28   | 50.37%     |  44.91%    | 

In the above table, LR refers to the Learning rate and $\Delta$ refers to a parameter of the loss function described below:
 $$loss = \frac{\sum_{i=1}^N max(0,| out_i - t_i | ^2 - \Delta)}{N} $$
 where  $out_i$ is the model's prediction, $t_i$ is the true label and $N$ is the number of training sequences.

Based on the obtained results, the following observations can be made:

1. The VA metric values for all implementations do not precisely match the numbers reported by the authors. Despite using the same hyperparameters as mentioned in the paper, it is possible that some unaccounted parameter, which the authors considered insignificant, could be influencing the discrepancy in the results.
2. However, we observe a similar trend to the authors, where the HM-LSTM outperforms the LSTM-only and FC-layers-only architectures. Although our achieved numbers are slightly lower than those reported by the authors, the accuracy of the loss function calculation in our implementation has been verified.
3. Possible reasons for the deviation in our results could be attributed to hyperparameter tuning or some undisclosed implementation details of the different architectures. While limited information is provided about the LSTM and FC-only layers, we assume minimal modifications are required from the original HM-LSTM architecture. Similar issues regarding the discrepancy in the numbers have been raised on GitHub, but the authors have not yet responded.
5. Hyperparameter tuning results indicate that the authors' chosen hyperparameters yield the best performance based on the testing we conducted. However, our experiments only focused on two parameters, namely the learning rate and theta. Further exploration, such as investigating the effect of dropout parameters through additional ablation studies, could provide deeper insights, but we have not pursued it at this stage.

In conclusion, although our achieved VA numbers do not precisely match those reported by the authors, we have obtained comparable results with a slightly lower performance. We provide possible reasons for this discrepancy and acknowledge the need for further investigation into the mentioned factors.

## Conclusion

The authors of the original paper introduce the RLDD dataset, a significant contribution to drowsiness detection research due to its larger size, encompassing nearly 30 hours of video data. Our implementation focuses on a baseline method that utilizes the temporal patterns of blinks to detect drowsiness. This method is designed to be computationally efficient and requires minimal storage. Experimental results reveal that the HM-LSTM approach outperforms other methods on the RLDD dataset, as indicated by two specific metrics proposed by the authors. However, it should be noted that our reproduced numbers do not precisely align with the results reported in the original paper. We attribute this discrepancy to the lack of detailed information provided regarding the hyperparameters used in their model.

## Challenges

The implementation of the code lacked sufficient comments and documentation, making it challenging to understand the input-output data formats for most functions. This resulted in extensive debugging efforts. Additionally, the absence of information about the implementation details of other architectures, such as the required dimensions for passing output from one layer to another, necessitated a trial-and-error approach. If the authors had provided more information, it would have significantly saved time and effort. We raised issues on the Github repository where the original code was uploaded and we received no response. This lack of information could also be a contributing factor to the discrepancy in our metric numbers compared to the original paper. It is possible that the authors implemented the other architectures differently, leading to variations in the results. 

## References

[1] Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). A realistic dataset and baseline temporal model for early drowsiness detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 0-0).

[2] Ngxande, M., Tapamo, J. R., & Burke, M. (2020, January). Detecting inter-sectional accuracy differences in driver drowsiness detection algorithms. In 2020 International SAUPEC/RobMech/PRASA Conference (pp. 1-6). IEEE.

[3] L. K. McIntire, R. A. McKinley, C. Goodyear, and J. P. McIntire. Detection of vigilance performance using eye blinks. Applied ergonomics, 45(2):354–362, 2014.

[4] U. Svensson. Blink behaviour based drowsiness detection. Technical report, 2004

[5] Q. Massoz, T. Langohr, C. Franc¸ois, and J. G. Verly. The ulg-multimodality drowsiness database (called drozy) and examples of use. In Applications of Computer Vision (WACV), 2016 IEEE Winter Conference on, pages 1–7. IEEE, 2016.

[6] E. Tadesse, W. Sheng, and M. Liu. Driver drowsiness detection through hmm based dynamic modeling. In Robotics and Automation (ICRA), 2014 IEEE International Conference on, pages 4003–4008. IEEE, 2014.

[7] B. Reddy, Y.-H. Kim, S. Yun, C. Seo, and J. Jang. Realtime driver drowsiness detection for embedded system using model compression of deep neural networks. In Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on, pages 438–445. IEEE, 2017.

[8] The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use", by Quentin Massoz, Thomas Langohr, Clémentine François, Jacques G. Verly, Proceedings of the 2016 IEEE Winter Conference on Applications of Computer Vision (WACV 2016), Lake Placid, NY, March 7-10, 2016.

[9] S. Park, F. Pan, S. Kang, and C. D. Yoo. Driver drowsiness detection system based on feature representation learning using various deep networks. In Asian Conference on Computer Vision, pages 154–164. Springer, 2016.

[10] Ching-Hua Weng, Ying-Hsiu Lai, Shang-Hong Lai, “Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network”, In Asian Conference on Computer Vision Workshop on Driver Drowsiness Detection from Video, Taipei, Taiwan, Nov. 2016.


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

