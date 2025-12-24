Pytorch Implementation of [[SIGGRAPH Asia 2025] ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction](https://arxiv.org/abs/2505.20431), [Qimin Chen](https://qiminchen.github.io/), [Yuezhi Yang](https://yyuezhi.github.io/), [Wang Yifan](https://yifita.netlify.app/), [Vladimir G. Kim](http://www.vovakim.com/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), [Hao Zhang](http://www.cs.sfu.ca/~haoz/), [Zhiqin Chen](https://czq142857.github.io/).

### [Paper](https://arxiv.org/abs/2505.20431)  |  [Project page](https://qiminchen.github.io/artdeco/)

<img src='teaser.jpg' />

## Test (Web UI)
1. Download the webUI code from here: [Google Drive](https://drive.google.com/file/d/1wxpQ2LQsEytuZVL4GOu1tur46FGRZCQv/view?usp=sharing) (We include 33 pre-trained weights corresponding to 33 text prompts)
2. Unzip (it does not matter where you unzip it)
3. Run (make sure you have the required dependencise installed, please refer to Dependencies for more details)
   ```
   cd web-demo
   python app.py
   ```
Note: This is the easiest way to test the model. If you want to export the textured mesh, please refer to [threestudio](https://github.com/threestudio-project/threestudio#export-meshes). 

## Citation
If you find our work useful in your research, please consider citing (to be updated):

	@inproceedings{chen2025artdeco,
	  title={ART-DECO: Arbitrary Text Guidance for 3D Detailizer Construction},
	  author={Chen, Qimin and Yang, Yuezhi and Wang, Yifan and Kim, Vladimir G and Chaudhuri, Siddhartha and Zhang, Hao and Chen, Zhiqin},
	  booktitle={SIGGRAPH Asia 2025 Conference Papers},
	  year={2025},
	}

## Dependencies
Requirements:
