# Clustering-Guided-Incremental-Learning-of-Tasks

![image](https://user-images.githubusercontent.com/47030528/116374735-89a06000-a849-11eb-8ac7-57221ee0e181.png)

This is an official Pytorch implementation of Clustering-guided Incremental Learning of Tasks. For details about this paper please refer to the paper [Clustering-Guided Incremental Learning of Tasks](https://ieeexplore.ieee.org/abstract/document/9334003) 

<br>


## Dependencies
    Python>=3.6
    PyTorch>=1.0
<br>


## Citing Paper

    @inproceedings{kim2021clustering,
    title={Clustering-Guided Incremental Learning of Tasks},
    author={Kim, Yoonhee and Kim, Eunwoo},
    booktitle={2021 International Conference on Information Networking (ICOIN)},
    pages={417--421},
    year={2021},
    organization={IEEE}
    }
    
<br>    
    

## Experiment (6 fine-grained image tasks)

|               |   ImageNet   | StanfordCars |     MNIST    |     CUBS     |    Sketch    |    Flowers   |
|:-------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| Ours      |    76.16     |    88.51     |    99.66     |    79.18     |    74.95     |    88.14     |
| Individual Training      |    76.16     |    61.56     |    99.87     |    65.17     |    75.40     |    59.73     | 

<br>
## Clustering Result
![image](https://user-images.githubusercontent.com/47030528/116376361-244d6e80-a84b-11eb-9fe2-60ea9c2bcf8c.png)

<br>
## Contact
Please feel free to leave comments to [YoonheeKim](https://github.com/Yooon-hee2)(yoooni.kim2@gmail.com)
