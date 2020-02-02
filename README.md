# Scene-Classification

The data is from Kaggle [intel_data_scene](https://www.kaggle.com/dipam7/intel-data-scene), You can download the full dataset by click [HERE](https://drive.google.com/file/d/1Hdi3mKpCEmY-vQz4mVT0z1Whqpg7JGBf/view?usp=sharing), and put them into the directory `/data`.

Authors: Xinneng XU, Ziheng LI

Automatic scene classification (sometimes referred to as scene recognition, or scene analysis) is a longstanding research problem in computer vision, which consists of assigning a label such as 'beach', 'bedroom', or simply 'indoor' or 'outdoor' to an image presented as input, based on the image's overall contents. 

This challenge will focus on the scene classification, which is to learn a model then use the model to classifier the image into a class automately. 

#### Set up for RAMP

Open a terminal and

1. install the `ramp-workflow` library (if not already done)

   ```
   $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
   ```

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](starting_kit.ipynb).

### Submissions

Execute

```
ramp_test_submission
```

to try the starting kit submission. You can also use

```
ramp_test_submission --submission=other_dir
```

to test any other submission located in a directory in the submission directory.