# Sleep-Robustness

Code used in the paper, "Biologically inspired sleep algorithm for increased generalization and adversarial robustness in deep neural networks"

Tadros, T., Krishnan, G., Ramyaa, R., & Bazhenov, M. (2019, September). Biologically inspired sleep algorithm for increased generalization and adversarial robustness in deep neural networks. In International Conference on Learning Representations.

The Attacks folder has code for setting up various adversarial attacks.

The code used to run the sleep algorithm is in the Sleep folder.

The neural network library is taken from another repository based on the following paper:
Palm, R. B. (2012). Prediction as a candidate for learning deep hierarchical models of data. Technical University of Denmark, 5.

To run the generalization test (blur and Gaussian noise), run the "compute_generalization_acc_4defenses_test.m" script.

To run the adversarial tests, you can run any of the "save_'attack'_results.m" scripts which will generate adversarial attacks based on the defenses tested in the paper.
