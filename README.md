# On the Sample Complexity of Learning from Pairwise Comparisons with Features

### Synthetic Experiments
To replicate results on a cluster with slurm, first adjust ```singe.bash```.
Then, run:
```sh
mkdir /home/$USER/Res-Synth/
sbatch main.bash
python src/figures_res_synth.py
```

### Sushi Experiments
To replicate results, first download  [Sushi Dataset] (sushi3-2016.zip) and extract to ```/home/$USER/Data/```.
Then, run:
```sh
mkdir /home/$USER/Res-Sushi/
python src/data.py
python sushi.py 1
python sushi.py 2
python src/figures_res_sushi.py
```

Please consider citing [our paper] if you use this repository.

[our paper]: <https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=on+the+sample+complexity+of+learning+from+pairwise+comparisons+with+features&btnG=>
[Sushi Dataset]: <http://www.kamishima.net/sushi/>
