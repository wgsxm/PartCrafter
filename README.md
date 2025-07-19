# PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers

<h4 align="center">

[Yuchen Lin<sup>*</sup>](https://wgsxm.github.io), [Chenguo Lin<sup>*</sup>](https://chenguolin.github.io), [Panwang Pan<sup>†</sup>](https://paulpanwang.github.io), [Honglei Yan](https://openreview.net/profile?id=~Honglei_Yan1), [Yiqiang Feng](https://openreview.net/profile?id=~Feng_Yiqiang1), [Yadong Mu](http://www.muyadong.com), [Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05573-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.05573)
[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://wgsxm.github.io/projects/partcrafter)
[<img src="https://img.shields.io/badge/YouTube-Video-red" alt="YouTube">](https://www.youtube.com/watch?v=ZaZHbkkPtXY)
[![Model](https://img.shields.io/badge/🤗%20Model-PartCrafter-yellow.svg)](https://huggingface.co/wgsxm/PartCrafter)
[![License: MIT](https://img.shields.io/badge/📄%20License-MIT-green)](./LICENSE)

<p align="center">
    <img width="90%" alt="pipeline", src="./assets/teaser.png">
</p>

</h4>

This repository contains the official implementation of the paper: [PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://wgsxm.github.io/projects/partcrafter/). 
PartCrafter is a structured 3D generative model that jointly generates multiple parts and objects from a single RGB image in one shot. 
Here is our [Project Page](https://wgsxm.github.io/projects/partcrafter).

Feel free to contact me (linyuchen@stu.pku.edu.cn) or open an issue if you have any questions or suggestions.


## 📢 News
- **2025-07-13**: PartCrafter is fully open-sourced 🚀.
- **2025-06-09**: PartCrafter is on arXiv. 

## 📋 TODO
- [x] Release inference scripts and pretrained checkpoints. 
- [x] Release training code and data preprocessing scripts. 
- [ ] Provide a HuggingFace🤗 demo.
- [ ] Release preprocessed dataset. 

## 🔧 Installation
Go to the project root directory:
```
python -m venv venv (activate venv)
```
From venv:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install deepspeed-0.17.2+a0bb346-py3-none-any.whl
pip install torch_cluster-1.6.3+pt27cu128-cp311-cp311-win_amd64.whl
```
Example startup args (models download automaticall)
```
python inference_partcrafter.py --image_path assets/images/np3_2f6ab901c5a84ed6bbdf85a67b22a2ee.png --num_parts 3 --tag robot
```
## 😊 Acknowledgement
We would like to thank the authors of [DiffSplat](https://chenguolin.github.io/projects/DiffSplat/), [TripoSG](https://yg256li.github.io/TripoSG-Page/), [HoloPart](https://vast-ai-research.github.io/HoloPart/), and [MIDI-3D](https://huanngzh.github.io/MIDI-Page/) 
for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation. 


## 📚 Citation
If you find our work helpful, please consider citing:
```bibtex
@misc{lin2025partcrafter,
  title={PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers}, 
  author={Yuchen Lin and Chenguo Lin and Panwang Pan and Honglei Yan and Yiqiang Feng and Yadong Mu and Katerina Fragkiadaki},
  year={2025},
  eprint={2506.05573},
  url={https://arxiv.org/abs/2506.05573}
}
```

## 🌟 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=wgsxm/PartCrafter&type=Date)](https://www.star-history.com/#wgsxm/PartCrafter&Date)
