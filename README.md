# DarkFeat

DarkFeat: Noise-Robust Feature Detector and Descriptor for Extremely Low-Light RAW Images (AAAI2023 Oral)

<img src="./fig/fig.gif" alt="darkfeat demo">

### Installation

```shell
git clone git@github.com:THU-LYJ-Lab/DarkFeat.git
cd DarkFeat
pip install -r requirements.txt
```

[Pytorch](https://pytorch.org/) installation is machine dependent, please install the correct version for your machine.

### Demo

```shell
python ./demo_darkfeat.py \
	--input /path/to/your/sequence \
	--output_dir ./output \
	--resize 960 640 \
	--model_path /path/to/pretrained/weights
```

Sample raw image sequences and pretrained weights can be downloaded from [here](https://drive.google.com/drive/folders/1zkUCsBVEmQcPZPhsEUymA5GIvAzi12hD?usp=sharing).

Note that different pytorch and cuda versions may cause different model output results, and the output matches may differ from those shown in the gif. The results are tested in python 3.6, PyTorch 1.10.2 and cuda 10.2.

### Evaluation

1. Download [MID](https://github.com/Wenzhengchina/Matching-in-the-Dark) Dataset.

2. Preprocessing the data in MID dataset, you can choose whether to enable histogram equalization or not:

   ```shell
   python raw_preprocess.py --dataset_dir /path/to/MID/dataset
   ```

3. Extract the keypoints and descriptors, followed by a nearest neighborhood matching:

   ```shell
   python export_features.py \
     --model_path /path/to/pretrained/weights \
     --dataset_dir /path/to/MID/dataset
   ```

4. Estimate the pose through corresponding keypoint pairs:

   ```shell
   python pose_estimation.py --dataset_dir /path/to/MID/dataset
   ```

5. Finally collect the results of pose estimation errors:

   ```
   python read_error.py
   ```

### Training from scratch

We use [GL3D](https://github.com/lzx551402/GL3D) as our source training-use matching dataset. Please follow the [instructions](https://github.com/lzx551402/GL3D) to download and unzip all the data (including GL3D group and tourism group).

Then using the preprocessing code provided by ASLFeat to generate matching informations:

```shell
git clone https://github.com/lzx551402/tfmatch
# please edit the GL3D path in the shell script before executing.
cd tfmatch
sh train_aslfeat_base.sh
```

To launch the training, configure your training hyperparameters inside `./configs` and then run:

```shell
# stage1
python run.py --stage 1 --config ./configs/config_stage1.yaml \
	--dataset_dir /path/to/your/GL3D/dataset \
	--job_name YOUR_JOB_NAME
# stage2 
python run.py --stage 2 --config ./configs/config_stage1.yaml \
	--dataset_dir /path/to/your/GL3D/dataset \
	--job_name YOUR_JOB_NAME \
	--start_cnt 160000
# stage3
python run.py --stage 3 --config ./configs/config.yaml \
	--dataset_dir /path/to/your/GL3D/dataset \
	--job_name YOUR_JOB_NAME \
	--start_cnt 220000
```

### Acknowledgements

This project could not be possible without the open-source works from [ASLFeat](https://github.com/lzx551402/ASLFeat), [R2D2](https://github.com/naver/r2d2),  [MID](https://github.com/Wenzhengchina/Matching-in-the-Dark), [GL3D](https://github.com/lzx551402/GL3D), [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork). We sincerely thank them all.