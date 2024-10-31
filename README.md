# [NeurIPS 2024] MSAD - A Benchmark for Video Anomaly Detection
![image-20241026003821228](https://gitee.com/zhu-liyun2000/typora_imgs/raw/master/img/202410260043576.png)

This is a implementation of our work **"Advancing Video Anomaly Detection: A Concise Review and a New Dataset"** (Accepted by 2024 NeurIPS Dataset and Benchmark Track).  We provide code for benchmarking our new Multi-Scenario Video Anomaly Detection dataset.



We propose a new **Multi-Scenario Anomaly Detection (MSAD) dataset**, a high- resolution, real-world anomaly detection benchmark encompassing diverse scenarios and anomalies, both human and non-human-related. For more details and applying for our MSAD dataset, please refer to our project website: [Project Website](https://msad-dataset.github.io)





## Dataset Preparation

If you would like to access the original dataset in video format, please submit a request through our online application form [here](https://forms.microsoft.com/pages/responsepage.aspx?id=XHJ941yrJEaa5fBTPkhkN0_bcDHlPvFAiLdm3BQe86NURVI5RlRWODhYWVZYSzNCSlBROThBTEQzOC4u&route=shorturl).

We also offer extracted features (I3D and Video-Swin Transformer) available for all researchers. 
[Feature_download](https://anu365.sharepoint.com/:f:/s/msad-dataset/ElG2Uers-Y5GilV-JDHPc6UBaOcGK_vLk3VPzCIN8Oe4KA?e=cHddTR) (Next expiry date: 29 Nov 2024, due to institutional limitations)

**Note that the Dataset is to be used solely for academic and research purposes. Commercial use, reproduction, distribution, or sale of the Dataset or any derivative works is strictly prohibited.**





## Benchmarking on MSAD Dataset

For benchmarking weakly-supervised methods on our MSAD dataset, we follow **Evaluation Protocol ii** as described in our paper. Training and testing file list can be found in [feature download](https://anu365.sharepoint.com/:f:/s/msad-dataset/ElG2Uers-Y5GilV-JDHPc6UBaOcGK_vLk3VPzCIN8Oe4KA?e=cHddTR) link.

**Protocal ii:** Train on 360 normal and 120 abnormal videos, and test on 120 normal and 120 abnormal videos. During training, we only provide video-level annotations. This protocol is suitable for evaluating weakly-supervised methods trained with our video-level annotations.

Our MSAD dataset supports benchmarking with the following weakly-supervised methods:

- RTFM
- MGFN
- UR-DMU (TODO)





## Citation

If you find MSAD useful in your research, please consider citing our paper 📝

```markdown
@inproceedings{msad2024,
    title = {Advancing Video Anomaly Detection: A Concise Review and a New Dataset},
    author = {Liyun Zhu and Lei Wang and Arjun Raj and Tom Gedeon and Chen Chen},
    booktitle = {The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year = {2024}
} 
```



## Acknowledgement

This codebase is built on top of [RTFM](https://github.com/tianyu0207/RTFM), [MGFN](https://github.com/carolchenyx/MGFN.), [UR-DMU](https://github.com/henrryzh1/UR-DMU), and we thank the authors for their work.

