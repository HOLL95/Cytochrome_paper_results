# Code for reproduction on the paper "Recovering Biological Electron Transfer Reaction Parameters from Multiple Protein Film Voltammetric Techniques Informed by Bayesian Inference"
To install and run the code, you will require a distribution of linux, git, python3.6-3.10 (and the associated package manager pip), boost and cmake. If you do not have these programs they can be installed on e.g. Ubuntu using ```apt```. 
```
sudo apt-get update
sudo apt-get install git python3.10 pip3 cmake libboost-all-dev
```

Once these packages are installed, you can clone this repository and its submodules, 
i.e. 

```
git clone --recurse-submodules https://github.com/HOLL95/Cytochrome_paper_results
```


Finally, you can install this repository using ```pip``` from inside the direcotry
```
pip install -e .
```


Simulation code can be found in the ```/src``` directory, and code to generate 
each of the figures in the main body of the paper can be found in ```/figures```
