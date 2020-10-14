# Link-Prediction

Data Node Text information at: https://drive.google.com/file/d/1KWwJkwbffjjRIzpMvEzKQYdtjCdSs4VX/view?usp=sharing ==> to add in `data/raw/` folder


**Problem description**: Your task will be to predict links between pages in a subgraph of the French webgraph. The webgraph is a directed graph G(V, E) whose vertices correspond to the pages of the French web, and a directed edge connects page U to page V if there exists a hyperlink on page U pointing to page V. From the original subgraph, edges have been deleted at random. Given a set of candidate edges, your job is to predict which ones appeared in the original subgraph. Each node is associated with a text file extracted from the html of the corresponding webpage. Your solution can be based on supervised or unsupervised techniques or on a combination of both. You should aim for the maximum mean F1 score.

<img src="https://github.com/gui-marra/link-prediction/blob/master/data/images/ProblemDescriptionExample.PNG" width="600">

**Data**: As described above, the problem data is given in two ways. The first, based on some connections of the analyzed subgraph. Secondly, we have the text referring to all nodes of this graph.  

**Strategy**: The challenge of this problem consists in extracting the maximum amount of useful information from the two forms of data we have to finally apply a supervised learning classification model. In this problem the best results were achieved concatenating link prediction features (using the networkx library) with different text similarities (cosine, ts-ss) using a tf-idf embedding model.  

<img src="https://github.com/gui-marra/link-prediction/blob/master/data/images/Strategy.PNG" width="600">


## 1. Feature Generation

#### 1.1 Graph Features

Given two nodes in a graph it is possible to calculate some coefficients that are correlated with the probability of these being connected. One of these is the Jaccard Coefficient, which measures the ratio of the number of neighbors in common over the total number of neighbors. All definitions can be found in the [networkx library](https://networkx.github.io/documentation/stable/reference/algorithms/link_prediction.html) reference.

<img src="https://github.com/gui-marra/link-prediction/blob/master/data/images/Networkx.PNG" width="800">

**Node2Vec**: node2vec is an algorithm to generate vector representations of nodes on a graph. The node2vec framework learns low-dimensional representations for nodes in a graph through the use of random walks [14] through a graph starting at a target node. [3]


#### 1.2 Text Features 

**TF-IDF**: TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. [4] [5]

`TS-SS`(Triangle Area Similarity * Sector's Area Similarity) similarity: https://github.com/taki0112/Vector_Similarity 



## 2. Classification Model

#### 2.1 Ensemble Classifiers 
Ensemble methods are meta-algorithms that combine several machine learning techniques into one predictive model in order to decrease variance (bagging), bias (boosting), or improve predictions (stacking). For example using multiple `DecisionTrees`[7] to form a Random Forest. Methods used in this challenge: 

`RandomForestClassifier` [8],
`ExtraTreesClassifier` [9],
`BaggingClassifier` [10],
`AdaBoostClassifier` [11],
`GradientBoostClassifier` [12]


#### 2.2 TPOT
TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. Reference: http://epistasislab.github.io/tpot/


#### 2.3 XGBoost
`XGBoost`[13] stands for Extreme Gradient Boosting; it is a specific implementation of the Gradient Boosting method which uses more accurate approximations to find the best tree model. One of the main differences is while regular gradient boosting uses the loss function of our base model (e.g. decision tree) as a proxy for minimizing the error of the overall model, XGBoost uses the 2nd order derivative as an approximation, and applies regularization (L1 & L2), which improves model generalization.



## 3. Results

**All results are stored in the folder: `results/performances`.** 

To test the feature generation simulating the proposed problem, we split our dataset into training set and test set, where the features of the graph come solely from the training set. This way we decrease the overfitting since the distribution of our test set would be equivalent to the submission set. 

The evaluations were made in the following order:
1. Using only the networkx features with several classification models (ensemble classifiers and TPOT autoML tool). Here we realize that results are better with the non-scaled features. The f1-score was approximately 90% in test set and 88% in submission set (final score). 
2. With the increment of graph features using node2vec embedding and the same classification models. Here we notice that the final score was not increased by these features, regardless of the classifier. 
3. With the in addition of textual features using tf-idf and XGBoost classification model. We chose this classifier because of its good performance and training speed using GPU, which is essential given the size of the dataset of this problem. These modifications led the f1-score to approximately 93.4% in the test set and 92.7% in the final score.



## References


[1] D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf

[2] T. Zhou, L. Lu, Y.-C. Zhang. Predicting missing links via local information. Eur. Phys. J. B 71 (2009) 623. https://arxiv.org/pdf/0901.0553.pdf

[3] Grover, Aditya and Leskovec, Jure. "node2vec: Scalable feature learning for networks." Paper presented at the meeting of the Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, 2016. https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf

[4] Juan Ramos. Using tf-idf to determine word relevance in document queries, 1999. https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.1424&rep=rep1&type=pdf 

[5] Shahmirzadi, O., Lugowski, A., Younge, K.: Text similarity in vector space models:
a comparative study. In: 2019 18th IEEE International Conference On Machine
Learning And Applications (ICMLA). pp. 659–666. IEEE (2019). https://arxiv.org/pdf/1810.00664.pdf

[6] Le, Quoc V. and Mikolov, Tomas. "Distributed Representations of Sentences and Documents.." Paper presented at the meeting of the ICML, 2014. https://cs.stanford.edu/~quocle/paragraph_vector.pdf 

[7] Quinlan, J. R.. "Induction of Decision Trees." Machine Learning 1 (1986): 81--106. https://hunch.net/~coms-4771/quinlan.pdf

[8] Breiman, Leo. "Random Forests." Machine Learning 45 , no. 1 (2001): 5-32. https://www.stat.berkeley.edu/users/breiman/randomforest2001.pdf


[9] Geurts, Pierre, Ernst, Damien and Wehenkel, Louis. "Extremely randomized trees.." Mach. Learn. 63 , no. 1 (2006): 3-42. https://jasonphang.com/files/extratrees.pdf 

[10] Kotsiantis, Sotiris & Tsekouras, George & Pintelas, P.. (2005). Bagging Model Trees for Classification Problems.. 328-337. https://www.researchgate.net/publication/221565417_Bagging_Model_Trees_for_Classification_Problems

[11] Schapire, Robert. (2013). Explaining AdaBoost. 10.1007/978-3-642-41136-6_5. http://rob.schapire.net/papers/explaining-adaboost.pdf


[12] Natekin, Alexey & Knoll, Alois. (2013). Gradient Boosting Machines, A Tutorial. Frontiers in neurorobotics. 7. 21. 10.3389/fnbot.2013.00021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/pdf/fnbot-07

[13] Chen, Tianqi and Guestrin, Carlos. "XGBoost: A Scalable Tree Boosting System.." Paper presented at the meeting of the KDD, 2016. https://arxiv.org/pdf/1603.02754.pdf

[14] Spielman, Daniel A. (2018) "Random Walks on Graphs" Yale Class https://www.cs.yale.edu/homes/spielman/561/lect10-18.pdf 
