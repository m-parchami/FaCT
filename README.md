<p align="center">
<h1 align="center">
FaCT : Faithful Concept Traces for Explaining Neural Network Decisions
</h1>

<p align="center">
<a href="https://m-parchami.github.io"><strong>Amin Parchami-Araghi</strong></a>
·
<a href="https://sukrutrao.github.io"><strong>Sukrut Rao</strong></a>
·
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/jonas-fischer"><strong>Jonas Fischer</strong></a>
·
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele"><strong>Bernt Schiele</strong></a>
</p>
  
<h3 align="center">
NeurIPS 2025
</h3>
  
<h3 align="center">
<a href="https://arxiv.org/abs/2510.25512">Paper</a>
|
<a>Video (soon)</a>
</h3>
</p>

<img width="1826" height="686" alt="grafik" src="https://github.com/user-attachments/assets/ca85c466-c082-493a-8dd7-f6c35393ff59" />


# Setup
```bash
git clone https://github.com/m-parchami/FaCT.git
conda create --name fact python=3.10
conda activate fact
cd FaCT; pip install -r requirements.txt
```
You can then either activate run `conda activate fact` before every script, or replace the CONDA variable in the script and point it towards the installed binary.

# Training your own FaCT model

The training implementation is provided under `fact.training`. You can use `train.sh` for sample training scripts for different layers and architectures. The script has the commands for both local runs, as well as `slurm` jobs.

```bash
bash train.sh
```

For every new layer or architecture that you visit, you would need to run the train script once to generate the features, and then afterwards run again, but with the `--precomputed-feats` pointing to the path of the feature tensor. Currently the code is configured to keep the size of the stored dataset to be 170GB (it adjusts itself based on number of dimensions). If 170 GB is too much for your setup, simply reduce number of feature samples per image [here](https://github.com/m-parchami/FaCT/blob/826b5e002d320326676223f5bcc58fd551f64f94/fact/utils/utils.py#L570) or specify a low `--num_batches` when collecting training features. I would recommend reducing the number of samples per image, so that the whole dataset is covered. I will soon add a less memory-intensive implementation of features, where it is stored in smaller tensor chunks that can be loaded independently.


# Loading the released FaCT models

We have released our FaCT models on Github. These are per-model and per-layer SAE checkpoints (Bias-free TopK-SAE) that would need to be plugged in to their corresponding [B-cos-v2](https://github.com/B-cos/B-cos-v2?tab=readme-ov-file#model-zoo) checkpoints. All of our evaluation scripts take care of this. The correct B-cos-v2 checkpoint is downloaded based on the `--arch` argument that you pass.

# Evaluating the FaCT models

All of the sample commands for this section (i.e model performance evaluation, concept consistency evaluation, and visualizations) are provided under `analysis.sh`. For all the evaluation scripts, if you mention `--download v0.1` it would use our provided checkpoints, and if not, it would assume that you have trained your own and try to load it locally.

## Evaluating Performance, Collecting Statistics

The `fact.analysis.analysis` evaluates the model accuracy and also collects activation and contribution statistics of concepts (over the test set). The statistics are required for the other evaluations below.

## Evaluating Concept Consistency (C2-Score)

Using `fact.analysis.dino_consistency` you can measure how consistent the top-X % of the concepts are. Our metric uses DINOv2 features and upsamples them with [LoftUP](https://github.com/andrehuang/loftup). You can in principle replace both the model and up-sampler with other alternatives if you wish. 

## Visualizing Concepts and Decision Making
You can use `fact.analysis.plot_inference` to either visualize concepts or visualize the decision making, by measuring concept contributions. There are examples for both provided in the main function of `fact.analysis.plot_inference`. This code also relies on the statistics from the above, to know which images activate the concepts highest, in order to visualize them.


I'd more than happy to help you with any questions you have regarding the implementation or the paper :) Simply create an issue here or write me an email.


If you use our models, metric, or implementation, kindly cite us under following:
```
@inproceedings{
    parchami2025fact,
    title={Fa{CT}: Faithful Concept Traces for Explaining Neural Network Decisions},
    author={Amin Parchami-Araghi and Sukrut Rao and Jonas Fischer and Bernt Schiele},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
}
```

