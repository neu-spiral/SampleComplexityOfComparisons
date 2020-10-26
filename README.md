# On the Sample Complexity of Learning from Pairwise Comparisons with Features

### Synthetic Experiments
To replicate results on a cluster with slurm, first adjust ```single.bash``` and ```single_by_M.bash```.
Then, run:
```sh
mkdir ~/Res-Synth/
mkdir ~/Res-Synth-M/
sbatch main.bash
sbatch main_by_d.bash
sbatch main_by_lambda.bash
python src/figures_res_synth.py
python src/figures_res_synth_by_M.py
```
You can run ```local_main.bash``` etc. for testing code locally.

### Sushi Experiments
To replicate results, first download  [Sushi Dataset] (sushi3-2016.zip) and extract to ```~/Data/```.
Then, run:
```sh
mkdir ~/Res-Sushi/
python src/data.py
python sushi.py 1
python src/figures_res_sushi.py
```

Please consider citing [our paper] if you use this repository.

[our paper]: <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=on+the+sample+complexity+of+learning+from+pairwise+comparisons+with+features&btnG=>
[Sushi Dataset]: <http://www.kamishima.net/sushi/>
