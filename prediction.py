import sys, re, string
import nltk
from nltk.tokenize import word_tokenize
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, SQLContext, Row, functions, types
from pyspark.sql.functions import concat, col, lit
spark = SparkSession.builder.appName('wikipedia_popular').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.setLogLevel('WARN')


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
	punc_re = re.compile('[%s]' % re.escape(string.punctuation))

	line = url_re.sub(' ', line).strip().strip('/')
	line = re.sub(r'\b(\w+)( \1\b)+', r'\1', line)
	line = punc_re.sub(' ', line)
	line = regex.sub(' ', line)
	line = re.sub(r'(.)\1+', r'\1\1', line)

  
	return line

# Word tokenizer Function
def word_tokenize1(x):
	lowerW = x.lower()
	yield word_tokenize(lowerW)


def main(input1, input2):

	# Schema for ngram data file
	observation_schema = types.StructType([
		types.StructField('first', types.StringType(), False),
		types.StructField('last', types.StringType(), False),
		types.StructField('probability', types.FloatType(), False),
		types.StructField('ngram', types.IntegerType(), False),
		])

	# Schema for user input
	schema1 = types.StructType([
		types.StructField('sentence', types.StringType(), True),])
	
	# read the ngram file
	data = spark.read.csv(input1,schema=observation_schema)
	# read user input
	text = sc.textFile(input2)
	# do cleaning
	line = text.map(decontracted)
	line = line.map(keep_letters)
	# tokenize
	words = line.flatMap(word_tokenize1)
	# count the length of the given sentence
	count1 = words.map(lambda x: len(x))
	count = count1.collect()[0]
	flag = 0;

	# if number of given word is greater than or equal to 2 then look at tri-gram
	if count >= 2:
		sentence = words.map(lambda x: (x, x[count-2], x[count-1]))
		df1 = sqlContext.createDataFrame(sentence, ['first_sen', 'word1', 'word2'])
		df1 = df1.select('first_sen', concat(col('word1'), lit(' '), col('word2')).alias('sentence'))
		df_prob = data.join(df1, (data['first'] == df1['sentence']) & (data['ngram']==3))
		# If given tri-gram is not found than go to bi-gram 
		if df_prob.count() == 0:
			flag = 1;
		# otherwise show the result
		else:
			df_prob = df_prob.select('first_sen', 'last', 'probability').orderBy('probability', ascending=False)
			df_prob.show()
	
	# if number of input word is one then look at bi-gram
	if count == 1 or flag == 1:
		# if given input is one
		if flag == 0:
			sentence = words.map(lambda x: x)
			df1 = sqlContext.createDataFrame(sentence, ['sentence'])
		# if given input is >=2 but failed to find corresponding word
		else:
			sentence = words.map(lambda x: (x, x[count-1]))
			df1 = sqlContext.createDataFrame(sentence, ['first_sen', 'sentence'])

		df_prob = data.join(df1, (data['first'] == df1['sentence']) & (data['ngram']==2))

		# If out of vocabulary
		if df_prob.count() == 0:
			flag = 2;
		else:
			df_prob = df_prob.select('first_sen', 'last', 'probability').orderBy('probability', ascending=False)
			df_prob.show()

	# for out of vocabulary
	if flag == 2:
		df1 = data.groupBy('ngram').max('probability')
		df1 = df1.withColumnRenamed('max(probability)', 'probability')
		df_join = data.join(df1, (data['probability'] == df1['probability']) & (data['ngram'] == 1)) \
					  .select(data['first'], data['probability'], data['ngram'])

		df_join.show()



if __name__ == '__main__':
	input1 = sys.argv[1]
	input2 = sys.argv[2]
	main(input1, input2)