# Join The High Accuracy Club on ImageNet Dataset With A Binary Neural Network Ticket
### 1.Introduction

Binary neural networks are the extreme case of network quantization, which has long been thought of as a potential edge machine learning solution. However, the significant accuracy gap to the full-precision counterparts restricts their creative potential for mobile applications. In this work, we revisit the potential of binary neural networks and focus on a compelling but unanswered problem: how can a binary neural network achieve the crucial accuracy level (e.g., 80%) on ILSVRC-2012 ImageNet? We achieve this goal by enhancing the optimization process from three complementary perspectives: (1) We design a novel binary architecture BNext based on a comprehensive study of binary architectures and their optimization process. (2) We propose a novel knowledge-distillation technique to alleviate the counter-intuitive overfitting problem observed when attempting to train extremely accurate binary models. (3) We analyze the data augmentation pipeline for binary networks and modernize it with up-to-date techniques from full-precision models. The evaluation results on ImageNet show that BNext, for the first time, pushes the binary model accuracy boundary to 80.57% and significantly outperforms all the existing binary networks.

![Pipeline](https://user-images.githubusercontent.com/24189567/187410143-b9f80b79-e3bf-4516-b90e-6edc2e1e33e8.jpg)
* **Figure**: The architecture of our design, constructed based on an optimized MobileNet backbone and the proposed modules. “Processor” is the core binary convolution module, enhanced using channel-wise mix-to-binary attention branch, and “BN” represents batch normalization layer. The basic block is consisted of an element attention module and a binary feed forward network.

![Convolution Comparison](https://user-images.githubusercontent.com/24189567/204559496-1729c13d-4149-43b5-b674-d0e3df81a72a.jpg)
* **Figure**: Convolution module comparison. a) is the basic module of XNOR Net. b) is the basic module of Real2Binary Net. c) is the core convolution processor in our design with adaptive mix2binary attention.

## 2.Pretrained Modles
|Method | Top-1 Acc| Pretrained Models| 
|:---:    | :---:     | :---:               |
|BNext-T|  72.4 % |  [BNext-T](https://owncloud.hpi.de/s/jKjwDk35vVRPQN0)                  |  
|BNext-S|  76.1 % |  [BNext-S](https://owncloud.hpi.de/s/bHLM7lqfzm58kIW)                  |
|BNext-M|  78.3 % |  [BNext-M](https://owncloud.hpi.de/s/jU5m9v4ADsJKZsa)                  |
|BNext-L|  80.6 % |  [BNext-L](https://owncloud.hpi.de/s/zQHrlxiQ6XbjCbz)                  |

### 3.Loss Landscape Visualization
![image](https://user-images.githubusercontent.com/24189567/188886411-7a478445-913b-41da-8183-7ab25688aca4.png)
* **Figure**: 3D loss landscape visualization comparison.

![image](https://user-images.githubusercontent.com/24189567/188886373-e532b4a5-6863-4d41-8d15-d3a7e98ff6d8.png)
* **Figure**: 2D loss contour line visualization comparison.
 
## 4.Training Procedure
![Training Procedure](https://user-images.githubusercontent.com/24189567/204558527-04de1a26-bfce-4a16-87f9-f781b13988f7.jpg)
* **Figure**: The loss curve, accuracy curve and temperature curve during the optimization process 

![comparison_sota](https://user-images.githubusercontent.com/24189567/204547298-291194f3-ed7d-4b84-9c8b-dc45dfa668da.png)
* **Figure**: Comparison with the state of the art binary neural network on ImageNet dataset.

