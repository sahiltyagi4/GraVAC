# _GraVAC_: Adaptive Compression for Communication-Efficient Distributed DL Training

**Implementation of adaptive compression and related work of _GraVAC_ presented at IEEE International Conference on Cloud Computing (CLOUD), 2023, Chicago, Illinois, USA.**

_Distributed data-parallel (DDP) training improves overall application throughput as multiple devices train on a subset of data and aggregate updates to produce a globally shared model. 
The periodic synchronization at each iteration incurs considerable overhead, exacerbated by the increasing size and complexity of state-of-the-art neural networks. 
Although many gradient compression techniques propose to reduce communica- tion cost, the ideal compression factor that leads to maximum speedup or minimum data exchange remains an open-ended problem since it varies with the quality of compression, model size and structure, hardware, network topology and bandwidth. 
We propose **GraVAC**, a framework to dynamically adjust compression factor throughout training by evaluating model progress and assessing gradient information loss associated with compression. 
GraVAC works in an online, black-box manner without any prior assumptions about a model or its hyperparameters, while achieving the same or better accuracy than dense SGD (i.e., no compression) in the same number of iterations/epochs. 
As opposed to using a static compression factor, GraVAC reduces end-to-end training time for ResNet101, VGG16 and LSTM by 4.32×, 1.95× and 6.67× respectively. 
Compared to other adaptive schemes, our framework provides 1.94× to 5.63× overall speedup._

**ACCESS LINKS**
- [Link1](https://ieeexplore.ieee.org/document/10255012)
- [Link2](https://sahiltyagi4.github.io/files/gravac.pdf)

**RUNNING**

- Still needs implementation of multi-level compression in _GraVAC_.
- Go to ```scripts``` directory to execute scripts: contains baseline uncompressed training or static compression training in ```run_baseline.sh```.
[Accordion](https://arxiv.org/abs/2010.16248) executed by ```run_accordion.sh```. And _GraVAC_ can be launched by ```run_gravac.sh```.
- Contains implementations of TopK, DGC, RedSync and RandomK compressions.
- Models trained: ResNet101, VGG16 and LSTM on CIFAR10, CIFAR100 AND PTB dataset.

**CITATION**
- **_Bibtex_**: @article{Tyagi2023GraVACAC,
  title={GraVAC: Adaptive Compression for Communication-Efficient Distributed DL Training},
  author={Sahil Tyagi and Martin Swany},
  journal={2023 IEEE 16th International Conference on Cloud Computing (CLOUD)},
  year={2023},
  pages={319-329}}