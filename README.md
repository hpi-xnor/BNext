# Join The "80%+ Top-1 Accuracy" Club on ImageNet Dataset With A Binary Neural Network Ticket
Binary neural network as the extreme case of network quantization has long been regarded as a potential solution for edge machine learning. However, the long accuracy gap to popular full precision design limits the application of binary neural networks on mobile computation. In this paper, we re-explore the possibility of binary neural network by answering an important but still open question: How can binary neural network reach 80%+ Top-1 accuracy on ImageNet task? Specifically, we answer the question from three folds: 1) We explore an optimization-friendly architecture. It surpasses previous binary model with a flatter loss landscape and therefore easier to optimize. 2) We combine the proposed arcitecture with a novel sample efficient knowledge distillation technique, where the binary model is constrained to pay more attention to the knowledge that has a closer manifold to student. 3) We "modernize" the binary network optimization with well designed optimization tricks to align it with current modern 32-bit architectures optimization. Augmented by the carefully designed optimization pipeline, the proposed architecture reaches 80%+ Top-1 accuracy on ImageNet classification task and achieves obvious advantages to popular INT8 networks in terms of a well balance among accuracy, inference speed and energy efficiency.   

![Pipeline](https://user-images.githubusercontent.com/24189567/187410143-b9f80b79-e3bf-4516-b90e-6edc2e1e33e8.jpg)
* **Figure**: The architecture of our design, constructed based on an optimized MobileNet backbone and the proposed modules. “Processor” is the core binary convolution module, enhanced using channel-wise mix-to-binary attention branch, and “BN” represents batch normalization layer. The basic block is consisted of an element attention module and a binary feed forward network.

![Convolution Comparison](https://user-images.githubusercontent.com/24189567/187469742-8d8f2a9b-d217-41f9-a127-aa584fe72755.jpg)
* **Figure**: Convolution module comparison. a) is the basic module of XNOR Net. b) is the basic module of Real2Binary Net. c) is the core convolution processor in our design with adaptive mix2binary attention.

![image](https://user-images.githubusercontent.com/24189567/188886411-7a478445-913b-41da-8183-7ab25688aca4.png)
* **Figure**: 3D loss landscape visualization comparison.

![image](https://user-images.githubusercontent.com/24189567/188886373-e532b4a5-6863-4d41-8d15-d3a7e98ff6d8.png)
* **Figure**: 2D loss contour line visualization comparison.

**Training from scratch:**
```py
   cd ImageNet
   
   bash run_distributed_on_disk_a6k5.sh
 ```
 
## Training Procedure
![Training Procedure](https://user-images.githubusercontent.com/24189567/186159600-b7f4342f-b54a-443e-91dc-bb8a86f59107.png)
* **Figure**: Training Top-1/5 and Testing Top-1/5 accuracy during training process.


![Comparison with SOTA BNNs](https://user-images.githubusercontent.com/24189567/186159512-4b9277fb-6b6e-4530-a2e5-6a6f3bfaf046.png)
* **Figure**: Comparison with the state of the art binary neural network on ImageNet dataset.

## Computing Graph Visualization

Support architecture basic block visualization. 

Users can get the computation graph file using the ```tensorboard``` library, and the generated graph file will be stored in the ```runs``` dir:
```py
  python ArchComparision.py --arc BNext --inplanes 64 --out_planes 64
```

Or the user can use the pre-computed architecture graph in ```visualization/runs```.

After that, we can visualize the computing graph using the following command: 
```py
  tensorboard --logdir runs/ --bind_all
```

