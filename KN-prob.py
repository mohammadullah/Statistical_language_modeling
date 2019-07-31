
"""
Created on Sat Nov 17 23:00:37 2018

@author: Mohammad Ullah
"""

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, types, SQLContext, Row, functions
from pyspark.sql.functions import udf, array, concat, col, lit
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import NGram
from operator import add
from pyspark.sql.functions import *
import sys, re, string
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+



# Functon to get the body section from the reddit entry
def raddit_body(line):                                              

    return line['body']

# Function to expand the contractions
def decontracted(line):
    # specific
    line = re.sub(r"won't", "will not", line)
    line = re.sub(r"can\'t", "can not", line)

    # general
    line = re.sub(r"n\'t", " not", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"\'s", " is", line)
    line = re.sub(r"\'d", " would", line)
    line = re.sub(r"\'ll", " will", line)
    line = re.sub(r"\'t", " not", line)
    line = re.sub(r"\'ve", " have", line)
    line = re.sub(r"\'m", " am", line)

    return line


# Function to clean each line
def keep_letters(line):

    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')
    regex = re.compile('[^a-zA-Z]')
    regex1 = re.compile('(\\b[B-Zb-z] \\b|\\b [B-Zb-z]\\b)')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))

    line = url_re.sub(' ', line).strip().strip('/')
    line = re.sub(r'\b(\w+)( \1\b)+', r'\1', line)
    line = punc_re.sub(' ', line)
    line = regex.sub(' ', line)
    line = regex1.sub(' ', line)
    line = re.sub(r'(.)\1+', r'\1\1', line)  

  
    return line

# Word tokenizer Function
def word_tokenize1(x):
    lowerW = x.lower()
    yield word_tokenize(lowerW)


substract_v = udf(lambda z: z-0.75, types.FloatType())
udf_lambda = udf(lambda z: 0.75*z[1]/z[0], types.FloatType())


def main(inputs, output):

    # Load and map jason file
    # Use reddit corpus
    text = sc.textFile(inputs).map(json.loads)                  
    # Extract reddit body
    line = text.map(raddit_body)
    # Expand all contracted words
    line = line.map(decontracted) 
    # Remove unnecessary stuffs from string                
    line = line.map(keep_letters)
    # tokenize the corpus
    words = line.flatMap(word_tokenize1)
    
    ############################# Uni-gram ##################################
    ## Mono-gram generation
    monogram = words.flatMap(lambda x:(x[i] for i in range(0,len(x)-1)))
    # make key, value pair
    result = monogram.map(lambda x: (x, 1))
    # reduceby key and add
    result = result.reduceByKey(add)
    result = result.sortBy(lambda x: x[1], ascending=False)
    # make dataframe
    df = sqlContext.createDataFrame(result, ['word', 'frequency1'])
    # calculate ML probability
    df = df.crossJoin(df.select(functions.sum('frequency1').alias('sum_freq'))) \
           .withColumn('prob1', functions.col('frequency1') / functions.col('sum_freq'))  
    df1 = df.withColumn('ngram1', functions.lit(1))
    df11 = df1.withColumn('last', functions.lit(23))
    # Rename coulmns
    df11 = df11.withColumnRenamed('word', 'first') \
               .withColumnRenamed('prob1', 'prob') \
               .withColumnRenamed('ngram1', 'ngram')
    df11 = df11.select('first', 'last', 'prob', 'ngram')  
    
    ############################# Bi-gram ##################################

    # Bi-gram generation
    bigrams = words.flatMap(lambda x: zip(x, x[1:]))
    result = bigrams.map(lambda x: (x, 1))
    result = result.reduceByKey(add)
    #result = result.filter(lambda x: x[1] > 3)
    
    df = sqlContext.createDataFrame(result, ['words', 'frequency'])
    df = df.select('words.*', 'frequency')
    df = df.withColumnRenamed('_1', 'first').withColumnRenamed('_2', 'last')
    # Calulate sum of all first words
    df_sum = df.groupBy('first').agg({'frequency' : 'sum'}) \
               .withColumnRenamed('sum(frequency)', 'first_sum') \
               .withColumnRenamed('first', 'first1')
    # Join the sum coulumn
    df_join = df.join(df_sum, df['first'] == df_sum['first1']) \
                .select(df['first'], df['last'], df['frequency'], df_sum['first_sum'])
    # Calculate the first section of KN_probability with fixed discount 0.75
    df_prob = df_join.withColumn('prob', (substract_v(df_join['frequency']))/df_join['first_sum'])
    # Count the number of unique word pair
    df_count = df_prob.groupBy('first').count()
    # Join dataframe
    df_join1 = df_prob.join(df_count, df_prob['first'] == df_count['first']) \
                      .drop(df_count['first'])
    # Calculate the lambda 
    df_join1 = df_join1.withColumn('lambda1', udf_lambda(array('first_sum', 'count'))) \
                       .select('first', 'last', 'first_sum', 'frequency', 'prob', 'count', 'lambda1')
    # Join dataframe
    df1_2 = df_join1.join(df1, df_join1['last'] == df1['word']) \
                    .withColumn('p1_lambda', df_join1['lambda1']*df1['prob1'])
    # Calculate the final probability
    df1_2 = df1_2.withColumn('prob2', df1_2['prob'] + df1_2['p1_lambda'])
    df2 = df1_2.orderBy('first').select('first', 'last', 'prob2')
    df2 = df2.withColumnRenamed('prob2', 'prob')
    df2 = df2.withColumn('ngram', functions.lit(2))
    
    ############################# Tri-gram ##################################
    # No comment is given as it is same as bi-gram
    # Tri-gram generation
    trigrams = words.flatMap(lambda x: zip(x, x[1:], x[2:]))
    result = trigrams.map(lambda x: (x, 1))
    result = result.reduceByKey(add)
    
    df = sqlContext.createDataFrame(result, ['words', 'frequency'])
    df = df.select('words.*', 'frequency')
    df = df.withColumnRenamed('_1', 'first').withColumnRenamed('_2', 'second') \
           .withColumnRenamed('_3', 'last')
    df = df.select(concat(col('first'), lit(' '), col('second')).alias('first'), 'second', 'last', 'frequency')
    df_sum = df.groupBy('first').agg({'frequency' : 'sum'}) \
               .withColumnRenamed('sum(frequency)', 'first_sum') \
               .withColumnRenamed('first', 'first1')
    df_join = df.join(df_sum, df['first'] == df_sum['first1']) \
                .select(df['first'], df['second'], df['last'], df['frequency'], df_sum['first_sum'])
    df_join = df_join.withColumnRenamed('first', 'first3') \
                     .withColumnRenamed('second', 'second3') \
                     .withColumnRenamed('last', 'last3') \
                     .withColumnRenamed('frequency', 'frequency3') \
                     .withColumnRenamed('first_sum', 'first_sum3')

    df_prob = df_join.withColumn('prob3', (substract_v(df_join['frequency3']))/df_join['first_sum3'])
    df_count = df_prob.groupBy('first3').count()
    df_count = df_count.withColumnRenamed('count', 'count3')
    df_join1 = df_prob.join(df_count, df_prob['first3'] == df_count['first3']) \
                      .drop(df_count['first3'])
    df_join1 = df_join1.withColumn('lambda3', udf_lambda(array('first_sum3', 'count3'))) \
                       .select('first3', 'last3', 'second3', 'first_sum3', 'frequency3', 'prob3', 'count3', 'lambda3')

    df2_3 = df_join1.join(df1_2, (df_join1['last3'] == df1_2['last']) & (df_join1['second3'] == df1_2['first'])) \
                    .withColumn('p2_lambda', df_join1['lambda3']*df1_2['prob2'])

    df2_3 = df2_3.withColumn('prob33', df2_3['prob3'] + df2_3['p2_lambda'])


    df3 = df2_3.orderBy('first3').select('first3', 'last3', 'prob33')
    df3 = df3.withColumnRenamed('first3', 'first') \
             .withColumnRenamed('last3', 'last') \
             .withColumnRenamed('prob33', 'prob')
    df3 = df3.withColumn('ngram', functions.lit(3))
 
    df_final = df3.union(df2)
    df_final = df_final.union(df11)

    df_final.write.csv(output)

if __name__ == '__main__':
    conf = SparkConf().setAppName('reddit averages')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.setLogLevel('WARN')
    assert sc.version >= '2.3'  # make sure we have Spark 2.3+
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)