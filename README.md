# Statistical Language Modeling

This is a statistical language modeling project using classical Markov Chain and Kenser-Ney (KN) smoothing. The main challenge of this project is not the modeling algorithm but rather the implementation of the algorithm in pyspark.


## Steps to generate probability and predict next word

Step-1: Run KN-prob.py on reddit dataset, this will generate an output file with ngrams and probability

 	${SPARK_HOME}/bin/spark-submit KN-prob.py reddit-3 output

Step-2: Run the prediction.py to  get the probability with predicted words on the terminal. You need to use two input folders. one from the output of the previous step and second one will contain a user input file with given text.
	
	${SPARK_HOME}/bin/spark-submit prediction.py output/ sentence/ 




