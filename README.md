Pytorch Implementation of [[SIGGRAPH Asia 2025] ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction](https://arxiv.org/abs/2505.20431), [Qimin Chen](https://qiminchen.github.io/), [Yuezhi Yang](https://yyuezhi.github.io/), [Wang Yifan](https://yifita.netlify.app/), [Vladimir G. Kim](http://www.vovakim.com/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), [Hao Zhang](http://www.cs.sfu.ca/~haoz/), [Zhiqin Chen](https://czq142857.github.io/).

### [Paper](https://arxiv.org/abs/2505.20431)  |  [Project page](https://qiminchen.github.io/artdeco/)

<img src='teaser.jpg' />

## Testing (Web UI)
1. Download the webUI code from here: [Google Drive](https://drive.google.com/file/d/1wxpQ2LQsEytuZVL4GOu1tur46FGRZCQv/view?usp=sharing) (We include 33 pre-trained weights corresponding to 33 text prompts)
2. Unzip (it does not matter where you unzip it)
3. Run (make sure you have the required dependencies installed, please refer to Dependencies for more details)
   ```
   cd web-demo
   python app.py
   ```
Note: We did not write testing code. Instead, we built this Web UI for testing. This is the easiest way to test the trained model. If you want to export the textured mesh, please refer to [threestudio](https://github.com/threestudio-project/threestudio#export-meshes). 

Note: To test the model you trained, since threestudio save everything in `last.ckpt`, please run the below command to clean up the `last.ckpt` and move it to the ckpts folder for testing. Please update the `index.html` and `PROMPT_TO_MODEL_PATH` in the `volume_generator.py` accordingly.

## Citation
If you find our work useful in your research, please consider citing (to be updated):

	@inproceedings{chen2025artdeco,
	  title={ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction},
	  author={Chen, Qimin and Yang, Yuezhi and Wang, Yifan and Kim, Vladimir G and Chaudhuri, Siddhartha and Zhang, Hao and Chen, Zhiqin},
	  booktitle={SIGGRAPH Asia 2025 Conference Papers},
	  year={2025},
	}

## Dependencies
This project is built upon [threestudio](https://github.com/threestudio-project/threestudio#export-meshes) and [MVDream-threestudio](https://github.com/threestudio-project/threestudio#export-meshes). Please follow [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio?tab=readme-ov-file#installation) for the required env installation.

## Data

Please download the data from [Google Drive](https://drive.google.com/file/d/1_BQuHGMVEkPVFL_JoEYhgdR5Y5VKnL3X/view?usp=sharing) and put them in `/threestudio/data/`.

## Training

We use two-stage training described in Sec. 3.4 of the paper for better structure generalization. This requires a single coarse voxel shape of the same shape category described in the text prompt. However, **if you do not have a single coarse voxel shape**, you can skip the first stage of training and simply follow the **Single-stage training** to train the model.

### Two-stage training

1. First, change the data path of the single coarse voxel shape in https://github.com/qiminchen/ART-DECO/blob/949ec8ff004e2baef6ea443d8facebf605cc1fbc/threestudio/models/geometry/voxel_grid_single.py#L68
2. Run below for the first-stage training
   ```
   python launch.py --config configs/mvdream-artdeco-sd21-single.yaml \
                    --train \
                    --gpu 0 \
                    system.geometry.category="03001627" \
                    system.prompt_processor.prompt="a Scandinavian-style chair with a clean design and soft fabric padding" \
   ```
3. Then run below for the second-stage training
   ```
   python launch.py --config configs/mvdream-artdeco-sd21.yaml \
                    --train \
                    --gpu 0 \
                    system.geometry.category="03001627" \
                    system.prompt_processor.prompt="a Scandinavian-style chair with a clean design and soft fabric padding" \
                    resume=Path/to/first/stage/ckpts/last.ckpt \
                    trainer.max_steps=50000
   ```

### Single-stage training

If you do not have a single coarse voxel shape, simply run
```
python launch.py --config configs/mvdream-artdeco-sd21.yaml \
				--train \
				--gpu 0 \
				system.geometry.category="03001627" \
				system.prompt_processor.prompt="a Scandinavian-style chair with a clean design and soft fabric padding" \
```
