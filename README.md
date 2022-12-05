# Join The High Accuracy Club on ImageNet With A Binary Neural Network Ticket

This is the official code of the paper [Guo, N., Bethge, J., Meinel, C. and Yang, H., 2022. Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket. arXiv preprint arXiv:2211.12933.](https://arxiv.org/abs/2211.12933).
### 1.Introduction

Binary neural networks are the extreme case of network quantization, which has long been thought of as a potential edge machine learning solution. However, the significant accuracy gap to the full-precision counterparts restricts their creative potential for mobile applications. In this work, we revisit the potential of binary neural networks and focus on a compelling but unanswered problem: how can a binary neural network achieve the crucial accuracy level (e.g., 80%) on ILSVRC-2012 ImageNet? We achieve this goal by enhancing the optimization process from three complementary perspectives: (1) We design a novel binary architecture BNext based on a comprehensive study of binary architectures and their optimization process. (2) We propose a novel knowledge-distillation technique to alleviate the counter-intuitive overfitting problem observed when attempting to train extremely accurate binary models. (3) We analyze the data augmentation pipeline for binary networks and modernize it with up-to-date techniques from full-precision models. The evaluation results on ImageNet show that BNext, for the first time, pushes the binary model accuracy boundary to 80.57% and significantly outperforms all the existing binary networks.

![Pipeline](https://user-images.githubusercontent.com/24189567/205317106-9a2448f7-116b-4f89-bdfb-c2d148287f52.jpg)
* **Figure**: The architecture of our design, constructed based on an optimized MobileNet backbone and the proposed modules. “Processor” is the core binary convolution module, enhanced using channel-wise mix-to-binary attention branch, and “BN” represents batch normalization layer. The basic block is consisted of an element attention module and a binary feed forward network.

![Convolution Comparison](https://user-images.githubusercontent.com/24189567/204559496-1729c13d-4149-43b5-b674-d0e3df81a72a.jpg)
* **Figure**: Convolution module comparison. a) is the basic module of XNOR Net. b) is the basic module of Real2Binary Net. c) is the core convolution processor in our design with adaptive mix2binary attention.

## 2.Pretrained Models
|Method | Top-1 Acc| Pretrained Models| 
|:---:    | :---:     | :---:               |
|BNext-T|  72.4 % |  [BNext-T](https://owncloud.hpi.de/s/jKjwDk35vVRPQN0)                  |  
|BNext-S|  76.1 % |  [BNext-S](https://owncloud.hpi.de/s/bHLM7lqfzm58kIW)                  |
|BNext-M|  78.3 % |  [BNext-M](https://owncloud.hpi.de/s/jU5m9v4ADsJKZsa)                  |
|BNext-L|  80.6 % |  [BNext-L](https://owncloud.hpi.de/s/zQHrlxiQ6XbjCbz)                  |
 
## 3.Training Procedure
![Training Procedure](https://user-images.githubusercontent.com/24189567/204558527-04de1a26-bfce-4a16-87f9-f781b13988f7.jpg)
* **Figure**: The loss curve, accuracy curve and temperature curve during the optimization process 

### 4. Feature Visualization
![Feature_Visualization_BNext_Tiny](https://user-images.githubusercontent.com/24189567/205326008-fde4e29b-e52a-4a90-81f9-88a45e736c8e.jpg)
* **Figure**: Visualizing the diversity of binary features in BNext-T model. The input image is resized as 1024x1024.

### 5. Reference
If you find our code useful for your research, please cite our paper as follows:
```
@article{guo2022join,
  title={Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket},
  author={Guo, Nianhui and Bethge, Joseph and Meinel, Christoph and Yang, Haojin},
  journal={arXiv preprint arXiv:2211.12933},
  year={2022}
}
```
