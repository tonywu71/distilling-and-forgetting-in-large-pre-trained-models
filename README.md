# Distilling and Forgetting in Large Pre-Trained Models 

Tony Wu, tw581@cam.ac.uk



## About this project

Repository containing the code for my dissertation on "Distilling and Forgetting in Large Pre-Trained Models" for the MPhil in Machine Learning and Machine Intelligence (MLMI) at the University of Cambridge.

Project supervisor: Mark Gales (mjfg100@cam.ac.uk)
Co-supervisors: Mengjie Qian (mq227@cam.ac.uk), Adian Liusie (al826@eng.cam.ac.uk)



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

7. On [Weight&Biases](http://wandb.ai), log in and create a new project e.g. `distilling-and-forgetting-in-large-pre-trained-models`

8. Edit the `configs/env_config.yaml` file accordingly. First, modify `WANDB_PROJECT` accordingly to the previous step. Then - if you are using Cambridge's HPC - the only other change needed is to replace `tw581` with your own Cambridge CRSid.



## Usage

The repository contains two important directories:

- `scripts`: with all the python scripts
- `run`: with all the SLURM job scripts

If the machine you are working on has a CUDA GPU, you can run the repository scripts directly from the `scripts` folder, e.g.:

```bash
python scripts/finetune_whisper_on_librispeech.py configs/whisper_small-librispeech_clean_100h.yaml
```

In general, it is recommended to submit SLURM jobs from the `run` folder:

```bash
sbatch finetune_whisper_on_librispeech.sh
```

You should be able to monitor both fine-tuning and evaluation directly from [Weight&Biases](http://wandb.ai).



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
