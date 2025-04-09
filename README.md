# FedPPO: Reinforcement Learning-Based Client Selection for Federated Learning With Heterogeneous Data

# required packages in the environment
conda env create -f fedppo_env.yaml

# parameters
| Paramater                         | Control                 | Default Value              | Choice                                           |
|-----------------------------------|-------------------------|----------------------------|--------------------------------------------------|
| iid & non-iid                     | `--iid`                 | `0` (0 for non-iid)        | `0`, `1`                                         |
| dataset                           | `--dataset`             | `cifar10`                  | `cifar10`, `cifar100`, `emnist`, `imdb`          |
| traing epoch                      | `--epoch`               | `100`                      |                                                  |
| cluser method                     | `--cluster_method`      | `hier`                     | `hier`, `gmm`, `dbscan`, `kmeans`, `no`          |
| device                            | `--gpu_id`              | `0`                        |  none for cpu device                             |
| user number                       | `--num_users`           | `50`                       | `25`, `50`, `100`, `200`                         |
|participation rate of users        | `--frac`                | `0.2`                      | `0.4`, `0.2`, `0.1`, `0.05`                      |


# Running the experiments
python ppo_remove_main.py --model=resnet10 --dataset=cifar10 --iid=0 --epochs=100 --level_n_system=0.0 --level_n_lowerb=0.0 --target_acc=0.9 --frac=0.2 --non_iid_prob_class=0.7 --alpha_dirichlet=10 --gpu_id=0

# Citation
If you find this project helpful, please consider to cite the following paper:
```
@ARTICLE{10909702,
  author={Zhao, Zheyu and Li, Anran and Li, Ruidong and Yang, Lei and Xu, Xiaohua},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={FedPPO: Reinforcement Learning-Based Client Selection for Federated Learning With Heterogeneous Data}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Noise measurement;Training;Data models;Accuracy;Noise;Adaptation models;Computational modeling;Servers;Distributed databases;Convergence;Reinforcement Learning;Federated Learning;Heterogeneous data},
  doi={10.1109/TCCN.2025.3547751}}
```
