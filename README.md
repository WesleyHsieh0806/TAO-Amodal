# TAO-Amodal

   Official Repository of Tracking Any Object Amodally.
   
   <!-- [**:paperclip: Paper Link**]() -->
   [**:orange_book: Project Page**](https://tao-amodal.github.io/)  | [**:paperclip: Paper Link**](https://arxiv.org/abs/2312.12433) | [**:pencil2: Citations**](#citations)
   
   <div align="center">
  <a href="https://tao-amodal.github.io/"><img width="95%" alt="TAO-Amodal" src="https://tao-amodal.github.io/static/images/webpage_preview.png"></a>
   </div>

</br>

  :pushpin: Leave a :star: to keep track of our updates.

---


  <h2> Table of Contents</h2>
  <ul>
    <li>
      <a href="#school_satchel-get-started">Get Started</a>
    </li>
    <li>
      <a href="#books-prepare-dataset">Download Dataset</a>
      <ul>
        <!-- <li><a href="#built-with">Built With</a></li> -->
      </ul>
    </li>
    <li>
      <a href="#artist-visualization">Visualization </a>
    </li>
    <li>
      <a href="#running-training">Training</a>
    </li>
    <li>
      <a href="#bar_chart-evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#citations">Citations</a>
    </li>
  </ul>



---

## :school_satchel: Get Started

Clone the repository
```bash
git clone https://github.com/WesleyHsieh0806/TAO-Amodal.git 
```

Setup environment
```bash
conda create --name TAO-Amodal python=3.9 -y
conda activate TAO-Amodal
bash environment_setup.sh
```

## :books: Prepare Dataset

1. Download our dataset following the instructions [here](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal).
2. The directory should have the following structure:

   ```
   TAO-Amodal
    ├── frames
    │    └── train
    │       ├── ArgoVerse
    │       ├── BDD
    │       ├── Charades
    │       ├── HACS
    │       ├── LaSOT
    │       └── YFCC100M
    ├── amodal_annotations
    │    ├── train/validation/test.json
    │    ├── train_lvis_v1.json
    │    └── validation_lvis_v1.json
    ├── example_output
    │    └── prediction.json
    ├── BURST_annotations
    │    ├── train
    │         └── train_visibility.json
    │    ...
    ```
   </p></details>
 
>    Explore more examples from our dataset [here](https://tao-amodal.github.io/dataset.html).

## :artist: Visualization



## :running: Training

## :bar_chart: Evaluation

1. Output tracker predictions as json.
The predictions should be structured as:
```bash
[{
    "image_id" : int,
    "category_id" : int,
    "bbox" : [x,y,width,height],
    "score" : float,
    "track_id": int,
    "video_id": int
}]
```

> We also provided an example output prediction json [here](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal/blob/main/example_output/prediction.json). Refer to this file to check the correct format.

2. Evaluate on TAO-Amodal
```bash
cd tools
python eval_on_tao_amodal.py --track_result /path/to/prediction.json \
                             --output_log   /path/to/output.log \
                             --annotation   /path/to/validation_lvis_v1.json
```
 


## Citations
``` bash
@misc{hsieh2023tracking,
    title={Tracking Any Object Amodally},
    author={Cheng-Yen Hsieh and Tarasha Khurana and Achal Dave and Deva Ramanan},
    year={2023},
    eprint={2312.12433},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

