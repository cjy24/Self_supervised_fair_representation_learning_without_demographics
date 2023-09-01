# Self_supervised_fair_representation_learning_without_demographics

Most literature on fairness assumes that the sensitive information, such as gender or race, is present in the training set, and uses this information to mitigate bias. However, due to practical concerns like privacy and regulation, applications of these methods are restricted. Also, although much of the literature studies supervised learning, in many real-world scenarios, we want to utilize the large unlabelled dataset to improve the model's accuracy. Can we improve fair classification without sensitive information and without labels? To tackle the problem, in this paper, we propose a novel reweighing-based contrastive learning method. The goal of our method is to learn a generally fair representation without observing sensitive attributes. Our method assigns weights to training samples per iteration based on their gradient directions relative to the validation samples such that the average top-k validation loss is minimized. Compared with past fairness methods without demographics, our method is built on fully unsupervised training data and requires only a small labelled validation set. We provide rigorous theoretical proof of the convergence of our model. Experimental results show that our proposed method achieves better or comparable performance than state-of-the-art methods on three datasets in terms of accuracy and several fairness metrics. Paper available at: https://proceedings.neurips.cc/paper_files/paper/2022/file/ad991bbc381626a8e44dc5414aa136a8-Paper-Conference.pdf

# Configuration

Please install all necessary packages via: pip install -r requirements.txt

# Usage

To run the experiments, please change the file directory accordingly. The CelebA dataset can be obtained at: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
# Citation

@article{chai2022self,
  title={Self-supervised fair representation learning without demographics},
  author={Chai, Junyi and Wang, Xiaoqian},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={27100--27113},
  year={2022}
}
