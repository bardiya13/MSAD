# MSAD - A Benchmark for Video Anomaly Detection
![image-20241026003821228](https://gitee.com/zhu-liyun2000/typora_imgs/raw/master/img/202410260043576.png)

This is a implementation of our work **"Advancing Video Anomaly Detection: A Concise Review and a New Dataset"** (Accepted by 2024 NeurIPS Dataset and Benchmark Track).  We provide code for our scene-adaptive method SA^2D and benchmarking our new Multi-Scenario Video Anomaly Detection dataset.



We propose a new Multi-Scenario Anomaly Detection (MSAD) dataset, a high- resolution, real-world anomaly detection benchmark encompassing diverse scenarios and anomalies, both human and non-human-related. For more details and applying for our MSAD dataset, please refer to our project website: [Project Website](https://msad-dataset.github.io)





## Dataset Preparation

Please apply for the dataset and complete the [Online form](https://forms.microsoft.com/pages/responsepage.aspx?id=XHJ941yrJEaa5fBTPkhkN0_bcDHlPvFAiLdm3BQe86NURVI5RlRWODhYWVZYSzNCSlBROThBTEQzOC4u&route=shorturl) here.

**Note that the Dataset is to be used solely for academic and research purposes. Commercial use, reproduction, distribution, or sale of the Dataset or any derivative works is strictly prohibited.**





## Benchmarking on MSAD Dataset

Method:

- RTFM
- MGFN (TODO)
- UR-DMU (TODO)





## Citation

If you find MSAD useful in your research, please consider citing our paper 📝

```markdown
@misc{zhu2024advancing,
      title={Advancing Video Anomaly Detection: A Concise Review and a New Dataset}, 
      author={Liyun Zhu and Lei Wang and Arjun Raj and Tom Gedeon and Chen Chen},
      year={2024},
      eprint={2402.04857},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.04857}, 
}
```



## Acknowledgement

This codebase is built on top of [FSAD](https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection) , [RTFM](https://github.com/tianyu0207/RTFM), and we thank the authors for their work.

