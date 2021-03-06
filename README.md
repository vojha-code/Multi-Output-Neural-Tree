# Multi-Output-Neural-Tree

Paper: https://arxiv.org/abs/2010.04524

We propose an algorithm and a new method to tackle the classification problems. We propose a multi-output neural tree (MONT) algorithm, which is an evolutionary learning algorithm trained by the non-dominated sorting genetic algorithm (NSGA)-III. Since evolutionary learning is stochastic, a hypothesis found in the form of MONT is unique for each run of evolutionary learning, i.e., each hypothesis (tree) generated bears distinct properties compared to any other hypothesis both in topological space and parameter-space. This leads to a challenging optimisation problem where the aim is to minimise the tree-size and maximise the classification accuracy. Therefore, the Pareto-optimality concerns were met by hypervolume indicator analysis. We used nine benchmark classification learning problems to evaluate the performance of the MONT. As a result of our experiments, we obtained MONTs which are able to tackle the classification problems with high accuracy. The performance of MONT emerged better over a set of problems tackled in this study compared with a set of well-known classifiers: multilayer perceptron, reduced-error pruning tree, naive Bayes classifier, decision tree, and support vector machine. Moreover, the performances of three versions of MONT's training using genetic programming, NSGA-II, and NSGA-III suggest that the NSGA-III gives the best Pareto-optimal solution.

<img src="https://github.com/vojha-code/Multi-Output-Neural-Tree/blob/main/results/MONT.png" alt="MONT" width="600">

<img src="https://github.com/vojha-code/Multi-Output-Neural-Tree/blob/main/results/Pareto_Front.png" alt="MONT" width="300" height="200"> <img src="https://github.com/vojha-code/Multi-Output-Neural-Tree/blob/main/results/Results.png" alt="MONT" width="300" height="200">


