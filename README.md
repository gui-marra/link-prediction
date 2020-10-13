# Link-Prediction

Data Node Text information at: https://drive.google.com/file/d/1KWwJkwbffjjRIzpMvEzKQYdtjCdSs4VX/view?usp=sharing ==> to add in `data/raw/` folder


**Problem description**: Your task will be to predict links between pages in a subgraph of the French webgraph. The webgraph is a directed graph G(V, E) whose vertices correspond to the pages of the French web, and a directed edge connects page U to page V if there exists a hyperlink on page U pointing to page V. From the original subgraph, edges have been deleted at random. Given a set of candidate edges, your job is to predict which ones appeared in the original subgraph. Each node is associated with a text file extracted from the html of the corresponding webpage. Your solution can be based on supervised or unsupervised techniques or on a combination of both. You should aim for the maximum mean F1 score.

**Data**: As described above, the problem data is given in two ways. The first, based on some connections of the analyzed subgraph. Secondly, we have the text referring to all nodes of this graph.  

**Strategy**: The challenge of this problem consists in extracting the maximum amount of useful information from the two forms of data we have to finally apply a supervised learning classification model. In this problem what brought the best results was concatenating link prediction features (using the networkx library) with different similarities (cosine, ts-ss) of texts using a tf-idf embedding model.  



## 1. Feature Generation

#### 1.1 Graph Features

#### 1.2 Text Features 




## 2. Classification Model


#### 2.1 Ensemble Classifiers

#### 2.2 TPOT

#### 2.3 XGBoost




## 3. Results

**All results are stored in the folder: `results/performances`.** 

To test the feature generation simulating the proposed problem, we split our dataset into training set and test set, where the features of the graph come solely from the training set. This way we decrease the overfitting since the distribution of our test set would be equivalent to the submission set. 

The evaluations were made in the following order:
1. Using only the networkx features with several classification models (ensemble classifiers and TPOT autoML tool). Here we realize that results are better with the non-scaled features. The f1-score was approximately 90% in test set and 88% in submission set (final score). 
2. With the increment of graph features using node2vec embedding and the same classification models. Here we notice that the performance was not increased by these features, regardless of the classifier. 
3. With the in addition of textual features using tf-idf and XGBoost classification model. We chose this classifier because of its good performance and training speed using GPU, which is essential given the size of the dataset of this problem. These modifications led the f1-score to approximately 93.4% in the test set and 92.7% in the final score.
4. results with the text features doc2vec. 





## References