# MARRO_Rhetorical-Role-Labeling
Rhetorical Role Labelling in a  multi-task learning setup uses label shift prediction as an auxiliary task.  

This repository contains the code for Rhetorical Role Labeling on India and UK Supreme Court data. We use various approaches, the most favorable being multi-task learning, which uses label shift prediction as an auxiliary task.

We can achieve SOTA performance on both datasets when compared to previous methods. 

In the following lines, we delineate the dataset (for now, we release a portion of it; please get in touch with the authors if you wish to gain access to the complete dataset). 

dataset folder has two folders:

##IN-dataset: Snippet of what the Indian Supreme Court data looks like. 
##UK-dataset: Snippet of what the UK Supreme Court data looks like. 

This work is a continuation of and builds on https://github.com/Law-AI/semantic-segmentation.

## 📄 Citation

If you use this dataset in your research, please cite our paper:

```bibtex
@article{bambroo2025marro,
  title={MARRO: Multi-Headed Attention for Rhetorical Role Labeling in Legal Documents},
  author={Bambroo, Purbid and Adhikary, Subinay and Bhattacharya, Paheli and Chakraborty, Abhijnan and Ghosh, Saptarshi and Ghosh, Kripabandhu},
  journal={arXiv preprint arXiv:2503.10659},
  year={2025}
}
