'''
Design Decisions:
(a) Removal of special characters, extra spaces, numbers, lower case letters and changing of character type from unicode to ascii has been made on initial basis 
(b) The dictionary of all the words from tweet has been made and the frequency of the words has been taken[1] in order to estimate the likelihood with each city 
(c) The stop words from the dictionary has been removed in order to get better predictions so that estimated location will be based on most frequent word in the tweet
(d) While calculating the likelihood of word being present in a particular city the instead of laplace smoothing i.e 1 as per given in [1] the very minimal factor has been taken i.e 0.1
    if the word is absent in the given city so that there will be no conflict with the word which has been present in the city with frequency 1 so the absent word does not make the half if
    smoothing of 1 is assigned which in turn also depends on the probabilities of other words present in tweet and may affect the overall probability
Experimentation:
Estimated accuracy of model with above given decisions = 66.8% (Current)
(a) Taking special characters in account decreases the accuracy of model by 2.6% than currently obtained.
(b) Taking all the words including stop words decreases the accuracy of model by 1.8% than currently obtained.
(c) Assuming laplace smoothing of 1 for not present words decreases the accuracy of model by 4% than currently obatained.
(d) Taking in account the unicode chars decreases the accuracy of model by 3.8% than currently obtained.
References:
[1] Multinomial Naive Bayes 'https://web.stanford.edu/class/cs124/lec/naivebayes.pdf'
[2] Get key with max value from dictionary 'https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary'
'''
import sys
import re
import copy
import math
import heapq 
	
#Taking input from the given file
def file_process(file_name, need_og_tweet):
	tweets = []
	og_tweet = []
	with open(file_name) as reader:
		for r in reader:
			if need_og_tweet:
				og_tweet.append(r.split(' '))
			r = r.decode('unicode_escape').encode('ascii', 'ignore')
			city = r.split(' ')[0]
			tweet = clean_text(r[r.index(' '):])
			r = city + ' ' + tweet
			tweets.append(r.strip().split())
	if need_og_tweet:
		return tweets, og_tweet
	else:
		return tweets

#Clean text
def clean_text(text):
	only_chars = '[^a-zA-Z\s]+' 
	text = re.sub(only_chars, '', text)
	text = re.sub(' +', ' ', text)
	text = text.replace('\r', '')
	text = text.replace('\n', '')
	return text.strip().lower()	

#get count of total tweets
def get_total_tweets(city_count):
	total_tweets = 0
	for i in city_count:
		total_tweets += city_count[i]
	return total_tweets

#city counter to keep track on number of tweets associated with cities
def city_counter(tweets):
	city_count = {}
	for r in tweets:
		city = r[0]
		if(city_count.has_key(city)):
			city_count[city] = city_count[city] + 1
		elif(city_count.has_key(city) == False):
			city_count[city] = 1
	return city_count

#method to calculate prior probabilities of all classes with respect to the total documents 
def cal_prior_prob(city_count, total_tweets):
	for i in city_count:
		city_prob[i] = float(city_count[i])/total_tweets

#making word dictionary for all words in all documents
def make_word_dict(cities):
	for r in cities:
		for word in r[1:]:
			if(word_dict.has_key(word) == False):
				word_dict[word] = 1
			else:
				word_dict[word] += 1 

#framing data to define the frequencies of all the words in training set
def define_freq():
	data = {}
	for i in city_count:
		data[i] = {}
	for r in range(0, len(cities)):
		for word in cities[r][1:]:
			if word_dict.has_key(word):
				if data[cities[r][0]].has_key(word):
					data[cities[r][0]][word] += 1
				else:
					data[cities[r][0]][word] = 1
			else:
				continue
	return data

#get total words in the given class
def get_total_words_gc(data):
	total = 0 
	for word in data:
		total += data[word]
	return total

#likelihood calc for each class
def cal_likelihood(data , word, given_class, words_in_class):
	num = 0
	len_of_dict = len(word_dict)
	total_words = 0
	if words_in_class.has_key(given_class):
		total_words = words_in_class[given_class]
	else:
		total_words = get_total_words_gc(data[given_class])
		words_in_class[given_class] = total_words
	if (data[given_class].has_key(word)):
		num += data[given_class][word]
	else:
		num += 0.1
	denom = total_words + len_of_dict
	return (float(num)/float(denom)), words_in_class

#calculate conditional probability for each word belongs to class
def cal_cond_prob(data, city_prob, cities):
	likelihood = {}
	words_in_class = {}
	for r in city_prob:
		likelihood[r] = {}
	for r in city_prob:
		for word in word_dict:
			word_likelihood, words_in_class = cal_likelihood(data, word, r, words_in_class)
			likelihood[r][word] = word_likelihood	
	return likelihood

#outputs the estimated label plus actual tweet to file
def output_prediction(estimated, og_tweet):
	with open(output_file, 'w') as outputfile:
		for i in range(0, len(og_tweet)):
			og_tweet[i].insert(0, estimated[i])
			outputfile.write(' '.join(map(str, 	og_tweet[i])))

#prediction of tweet locations
def predict(file_name):
	test_file, og_tweet = file_process(file_name, True)
	count_correct = 0
 	count_incorrect = 0
	estimated = []
	for tweet in test_file:
		pred_class = {}
		for city in city_prob:
			mul = city_prob[city]
			for word in tweet[1:]:
				if word_dict.has_key(word):
					mul = mul * likelihood_data[city][word]
				else:
					continue
			pred_class[city] = mul
			prediction = max(pred_class.iterkeys(), key = lambda k : pred_class[k])	#Ref[2]
		estimated.append(prediction)
		if prediction == tweet[0]:
 			count_correct += 1
 		else:
 			count_incorrect += 1
	output_prediction(estimated, og_tweet)	
	print 'Correctly classified', count_correct
	print 'Accuracy', (float(count_correct)/len(test_file))*100

#Removal of stop words and keeping most frequent words
def get_most_freq(word_dict):
	word_dict2 = copy.deepcopy(word_dict)
	for word in word_dict:
		if word in stop_words:
			del word_dict2[word]
	return word_dict2	

#gets the top five words associated with all 12 cities
def top_five_occ(likelihood_data):
	for city in likelihood_data:
		print city
		print heapq.nlargest(5, likelihood_data[city], key = likelihood_data[city].get)
		print

#program start
train_file = str(sys.argv[1])
test_file = str(sys.argv[2])
output_file = str(sys.argv[3])
stop_words = set(['all', 'whys', 'being', 'over', 'isnt', 'through', 'yourselves', 'hell', 'its', 'before', 'wed', 'with', 'had', 'should', 'to', 'lets', 'under', 'ours', 'has', 'ought', 'do', 'them', 'his', 'very', 'cannot', 'they', 'werent', 'not', 'during', 'yourself', 'him', 'nor', 'wont', 'did', 'theyre', 'this', 'she', 'each', 'havent', 'where', 'shed', 'because', 'doing', 'theirs', 'some', 'whens', 'up', 'are', 'further', 'ourselves', 'out', 'what', 'for', 'heres', 'while', 'does', 'above', 'between', 'youll', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'both', 'about', 'would', 'wouldnt', 'didnt', 'ill', 'against', 'arent', 'youve', 'theres', 'or', 'thats', 'weve', 'own', 'whats', 'dont', 'into', 'youd', 'whom', 'down', 'doesnt', 'theyd', 'couldnt', 'your', 'from', 'her', 'hes', 'there', 'only', 'been', 'whos', 'hed', 'few', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'on', 'but', 'you', 'hadnt', 'shant', 'mustnt', 'herself', 'than', 'those', 'he', 'me', 'myself', 'theyve', 'these', 'cant', 'below', 'of', 'my', 'could', 'shes', 'and', 'ive', 'then', 'wasnt', 'is', 'am', 'it', 'an', 'as', 'itself', 'im', 'at', 'have', 'in', 'id', 'if', 'again', 'hasnt', 'theyll', 'no', 'that', 'when', 'same', 'any', 'how', 'other', 'which', 'shell', 'shouldnt', 'our', 'after', 'most', 'such', 'why', 'wheres', 'a', 'hows', 'off', 'i', 'youre', 'well', 'yours', 'their', 'so', 'the', 'having', 'once', ''])
city_prob = {} 
cities = file_process(train_file, False)
city_count = city_counter(cities) 
total_tweets = get_total_tweets(city_count)

cal_prior_prob(city_count, total_tweets)

word_dict = {}
make_word_dict(cities)
word_dict = get_most_freq(word_dict)

df = define_freq()
likelihood_data = cal_cond_prob(df, city_prob, cities)

top_five_occ(likelihood_data)
predict(test_file)	
