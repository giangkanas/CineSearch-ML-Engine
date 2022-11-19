# information-retrieval-project
In this project, we have to create a search engine that is based on Elasticsearch API. This search engine has to decide the order of the results using Machine Learning techniques.


In the first task, we have to load the movies file to Elasticsearch. After that we have to write a second program that gets a string as an input and it returns a list of movies that fits in this search. The order of the movies has to come of the metric BM25.


In the second task, we have to write a program that asks for a User Id (integer) and a string as an input. In this program the search engine has to return a list of movies that fits in this search. The order of the movies has to come of a new metric that combines metric BM25, user's rating for the movie (if it' s available) and mean rating of the movie.


In the third task, we have to repeat the process that we followed in the previous task with one addition. For the movies that is not rated by the user (unavailable user's rating in the ratings dataset), we have to predict this user's rating with the following way. 

-First step, we have to compute the mean rating of every movie genre for every user.

-Second step, we have to cluster the users. For the clustering we have to use the mean ratings that are computed in the first step. We use K-Means for clustering.

-Third step, we compute the mean rating for each movie for each cluster

Now, in order to predict the missing rating, we can use the mean rating of the movie among the users of the same cluster.


In the fourth task, in order to predict the missing ratings we train a Neural Network for each user. The training data come of the movies that are rated by the user.
With the Neural Network, we predict what would be the ratings of the missing movies, if he would rate them.
In this task, we use techniques like Word Embeddings and One Hot Encoding.

