# Joint Inference of Multiple Graphs with Hidden Variales

The code in this repository implements the algorithm introduced in the paper [1], submited to ICASSP 2022.

The organization of the repository is as follows:
- **hidden_influence.m**: includes the code of the test case 1 in the paper.
- **samples_baselines.m**: includes the code of the test case 2 in the paper.
- **samples_real.m**: includes the code of the test case 3 in the paper.
- **opt**: contains the optimization algorithms used in the previous simulations. Include the proposed algorithm and other alternatives used as baselines.
- **utils**: contains some utility functions such as generating similar ER graphs or scripts to perform a grid search for the weights of the regularizers.
- **data**: contains the dataset employed in the test case 3.

[1] S. Rey, A. Buciulea, M. Navarro, S. Segarra, A, Marques. "Joint inference of multiple graphs with hidden variables from stationary graph signals".
