# Distilling and Forgetting in Large Pre-Trained Models 

Tony Wu, tw581@cam.ac.uk



## About this project

Repository containing the code for my dissertation on "Distilling and Forgetting in Large Pre-Trained Models" for the MPhil in Machine Learning and Machine Intelligence (MLMI) at the University of Cambridge.

Project supervisor: Mark Gales (mjfg100@cam.ac.uk)
Co-supervisors: Mengjie Qian (mq227@cam.ac.uk), Adian Liusie (al826@eng.cam.ac.uk)



## TL;DR

- Distilled state-of-the-art speech recognition Whisper. Achieved a −5% relative word error rate (WER) compared to vanilla fine-tuning for supervised distillation and a −17.81% relative WER for unsupervised fine-tuning.
- Implement continual learning solutions for fine-tuning multilingual Whisper on English. Achieved a −25% relative multilingual WER drop compared to vanilla fine-tuning.
- Coded from scratch using PyTorch, HuggingFace, and Weight&Biases.



## Abstract

> Large pre-trained models have become popular for deep learning applications because of their impressive performance. Moreover, they can be fine-tuned to perform specific tasks with relatively small amounts of data. However, many challenges remain to be tackled. Notably, the substantial number of parameters in these models makes them computationally expensive and slow at inference. Additionally, a separate issue arises when fine-tuning models to incorporate new tasks, as this process often leads to the forgetting of previously acquired knowledge. This is problematic as a robust and generic model is desirable. This dissertation investigates methods to solve these two issues. The application of these techniques is tested on Whisper, a state-of-the-art Automatic Speech Recognition (ASR) foundation model with a Transformer architecture.
>
> Firstly, the trade-off between model size and performance can be solved by knowledge distillation. Distillation is a machine learning training technique used to transfer knowledge from a large model (the *teacher*) to a smaller one (the *student*). At its core, knowledge distillation involves training the student to mimic the teacher’s output logits. This thesis explores two unsupervised distillation methods suited for sequence-to-sequence models: word-level distillation and K-best sequence-level distillation. The models are trained exclusively on AMI, a 100h-long English conversational dataset. Using Whisper tiny as the student and Whisper medium as the teacher, it is shown that naively using the raw teacher outputs for 1-best distillation gives a poor word error rate (WER) improvement from 27.74% to 27.30% for 1-best distillation on the AMI test set. Therefore, normalizing the teacher’s predictions and filtering out hallucinations prior to training are investigated to improve the distillation performance. In particular, filtering out the teacher’s transcriptions with high values of gzip compression ratio is demonstrated to be quite effective: while it removes only 0.08% of the examples and 0.26% of the audio from the training set, the subsequent 1-best yields a much more reasonable WER decrease from 27.74% to 22.80% on AMI test, i.e. a 17.80% relative improvement.
>
> Secondly, continual learning can be achieved using Elastic Weight Consolidation (EWC). By estimating the Fisher information matrix for the previous tasks, it is possible to measure the task-related importance of the model’s parameters, which EWC can *a fortiori* adequately regularize during fine-tuning. Furthermore, Task Alignment Consolidation (TAC) - a novel method introduced in this thesis - considers regularization from the prediction space to preserve the multilingual capabilities of Whisper without needing data from the previous tasks. Nonetheless, this method is shown to be inefficient. On the other hand, EWC proves to be an impressive candidate for continual learning. Not only does it manage to keep the relative WER increase for French transcription under 4% of the vanilla performance for the whole training, but it also significantly reduces forgetting for other non-English transcription tasks: compared to default fine-tuning, EWC achieves an average relative WER drop of 25.13% on the non-English datasets from Multilingual LibriSpeech.



## Installation

1. Clone the repository by running the following command in a bash terminal

   ```bash
   git clone https://github.com/tonywu71/distilling-and-forgetting-in-large-pre-trained-models
   ```

2. Navigate to the project directory using `cd distilling-and-forgetting-in-large-pre-trained-models/`

3. You should then checkout to the latest release (which you identify using `git tag` or on [GitHub](https://github.com/tonywu71/distilling-and-forgetting-in-large-pre-trained-models/releases)), run the command `git checkout <tag-name>` in your terminal. Replace `<tag-name>` with the actual tag name of the release you want to check out to.

4. Install the required dependencies using `pip install -r requirements.txt` (in a virtual environment if necessary)

5. Run the following to login to:

   ```bash
   huggingface-cli login
   ```

6. Run the following to login to wandb:

   ```bash
   wandb login
   ```

7. On [Weight&Biases](http://wandb.ai), log in and create 2 new projects: one for the training e.g `distilling_and_forgetting-training` and one the evaluation e.g. `distilling_and_forgetting-evaluation`.

8. Edit the `configs/env_config.yaml` file accordingly. First, modify `WANDB_PROJECT_TRAINING` and `WANDB_PROJECT_EVALUATION` accordingly to the previous step. Then - if you are using Cambridge's HPC - the only other change needed is to replace `tw581` with your own Cambridge CRSid.



## Usage

The repository contains two important directories:

- `scripts`: with all the python scripts
- `run`: with all the SLURM job scripts

If the machine you are working on has a CUDA GPU, you can run the repository scripts directly from the `scripts` folder, e.g.:

```bash
python scripts/finetune_whisper.py configs/whisper_small-librispeech_clean_100h.yaml
```

In general, it is recommended to submit SLURM jobs from the `run` folder:

```bash
sbatch run/finetune/multilingual/finetune_whisper_on_librispeech_100h.sh
```

You should be able to monitor both fine-tuning and evaluation directly from [Weight&Biases](http://wandb.ai).



## Examples

A non-exhaustive list of bash interactions is shown below. Note that you can easily create your own YAML config files based on the attributes of the `FinetuneConfig`, `DistilConfig`, `EWCFinetuneConfig`, and `TACFinetuneConfig` classes.

- Fine-tune Whisper `tiny` on LibriSpeech 100h:
  ```bash
  python scripts/finetune_whisper.py configs/finetune_configs/multilingual/librispeech/finetune_tiny-librispeech_100h.yaml
  ```

- Distil Whisper `medium`  into Whisper `tiny`:

  ```bash
  python scripts/distil_whisper.py configs/distil_configs/1_best/librispeech_100h/distil_1_best-medium_to_tiny-librispeech.yaml
  ```

- Compute and save the EWC parameters:

  ```bash
  python scripts/save_whisper_ewc_params.py \
      openai/whisper-tiny \
      --language french \
      --task transcribe \
      --dataset-name mls_french_diagnostic \
      --split train
  ```

- Fine-tune using EWC:

  ```bash
  python scripts/finetune_whisper.py \
      configs/finetune_ewc_configs/combined/finetune_ewc_tiny-ami_100h-combined-full.yaml \
      --ewc
  ```

- Fine-tune using TAC:

  ```bash
  python scripts/finetune_whisper.py \
      configs/finetune_tac_configs/small/finetune_tac_hpt-gamma_1e-1.yaml \
      --tac
  ```



## References

```bibtex
@misc{radfordRobustSpeechRecognition2022,
  title = {Robust {{Speech Recognition}} via {{Large-Scale Weak Supervision}}},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  year = {2022},
  month = dec,
  number = {arXiv:2212.04356}
}
```
