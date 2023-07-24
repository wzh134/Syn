# Concentration or distraction? A synergetic-based attention weights optimization method
*Zihao Wang, Haifeng Li, Lin Ma & Feng Jiang*

The attention mechanism empowers deep learning to a broader range of applications, but the contribution of the attention module is highly controversial. Research on modern Hopfield networks indicates that the attention mechanism can also be used in shallow networks. Its automatic sample filtering facilitates instance extraction in Multiple Instances Learning tasks. Since the attention mechanism has a clear contribution and intuitive performance in shallow networks, this paper further investigates its optimization method based on the recurrent neural network. Through comprehensive comparison, we find that the Synergetic Neural Network has the advantage of more accurate and controllable convergences and revertible converging steps. Therefore, we design the Syn layer based on the Synergetic Neural Network and propose the novel invertible activation function as the forward and backward update formula for attention weights concentration or distraction. Experimental results show that our method outperforms other methods in all Multiple Instances Learning benchmark datasets. Concentration improves the robustness of the results, while distraction expands the instance observing space and yields better results.

Paper available at [here](https://link.springer.com/article/10.1007/s40747-023-01133-0).

## Requirements

Codes were developed and tested on the following 64-bit operating system:

* Windows 11

The developing environment includes:

* Python 3.9.13
* Pytorch 1.12.0
* CUDA 11.6 with CUDNN

## Files

Files includes:

* mil_bags.py: Dataset import and preprocessing.
* syn.py: Realization of the network with Syn layer and SynPooling module.
* train_syn.py: Codes for network training.

## Usage

1. Put the dataset into ./mil_datasets/ folder. The supported datasets include
   * [Elephant, Fox, and Tiger](https://dl.acm.org/doi/10.5555/2968618.2968690)
   * [Musk 1 & 2](https://dl.acm.org/doi/10.1016/S0004-3702(96)00034-3)
   * [UCSB](https://bioimage.ucsb.edu/research/bio-segmentation)
   * [Web Recommendation](https://dl.acm.org/doi/10.1007/s10489-005-5602-z)
2. Run "train_syn.py --seed x --set yyy". "x" with range 0-255 controls the random partition seed of the dataset, and "yyy" is the abbreviation of the dataset. Abbreviation details and additional parameters for parallel training can be found in file ./train_syn.py
3. Adjust the search grid in the train_syn.py. Hyperparameters include
   * lr: learning rate
   * embed_layer_num: the layer number for input embedding
   * hidden_dim: the node number of the embedding layer
   * num_head: the head number of the SynPooling
   * embed_dim: the embed number of SynPooling's input embedding
   * step: the iteration steps for Syn layer
   * scale: the scaling factor of the query matrix

