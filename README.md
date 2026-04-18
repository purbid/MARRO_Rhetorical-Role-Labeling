# MARRO_Rhetorical-Role-Labeling
This repository focuses on **Rhetorical Role Labeling** in legal documents using a **multi-task learning framework**, where **label shift prediction** is incorporated as an auxiliary task.

We experiment with multiple approaches, with **multi-task learning** emerging as the most effective strategy.

---

## 🚀 Key Highlights

- 📊 Achieves **state-of-the-art (SOTA)** performance on both:
  - Indian Supreme Court dataset
  - UK Supreme Court dataset  
- 🧠 Utilizes **label shift prediction** as an auxiliary task to enhance performance  
- ⚖️ Designed specifically for **legal document understanding**

---

## 📂 Dataset

Currently, we release a **subset of the dataset**.  
If you are interested in accessing the **full dataset**, please contact the authors.

The `dataset/` directory contains:

- `IN-dataset/`  
  → Sample data from the **Indian Supreme Court**

- `UK-dataset/`  
  → Sample data from the **UK Supreme Court**

---

## 🔗 Related Work

This work builds upon and extends:  
👉 https://github.com/Law-AI/semantic-segmentation

---

## 📬 Note

For full dataset access or collaboration inquiries, feel free to reach out to the authors.
## 📄 Citation

If you use this dataset in your research, please cite our paper:

```bibtex
@article{bambroo2025marro,
  title={MARRO: Multi-Headed Attention for Rhetorical Role Labeling in Legal Documents},
  author={Bambroo, Purbid and Adhikary, Subinay and Bhattacharya, Paheli and Chakraborty, Abhijnan and Ghosh, Saptarshi and Ghosh, Kripabandhu},
  journal={arXiv preprint arXiv:2503.10659},
  year={2025}
}
