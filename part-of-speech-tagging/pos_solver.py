#!/usr/bin/env python
 
###################################
# (Based on skeleton code by D. Crandall)
####
'''
HMM Abstraction:
- States: POS tags corresponding to each word
- Obsevables: words of a given sentence
- Intial Probaility: Probability of a state (POS tag) starting a sentence. We calculate it by taking the
                   count of each POS at the start of sentence and divided it with the total number of sentences
- Transition Probability: Probability of transition from state 1 to state 2; we calculated it by taking the count of transition
                   from state 1 to state 2 and divided by the total transitions of state 1
- Emission Probability: Probability of a word for a given POS. We calculated it by counting the occurances of a word for a given 
                   POS and divided by the total occurances of that POS
- State Probability: Probability of each state (POS tag). We calculated it counting the total occurances of a POS divided by
                   sum of occurances of all POS tags
- Posterior: Probaility of a state given a sentence =  P(S1 ... Sn | W1 ... Wn). Using Bayes rule and
             naive bayes assumption and by first order markow assumption, we can write it as 
             P(W1 | S1) ... P(Wn | Sn) P(S1) P(S2|S1) ... P(Sn | Sn-1)/ P(W1...Wn), where P(Wi | Si) is the emission
             probability and P(Si+1|Si) is the transition probability. We have ignored the denominator here.
- Program workflow:
	- In the initial stage of training the counts of word, state, start words, emission (word | state), transition(current state | previous state) have been taken and then probabilities
	  for the same have been calculated.
	- After calculating above given probabilities we've used them to predict the state(POS) corresponding to each word in the given sentence for
	  each algorithms(Simplified, Variable Elimination, Viterbi).
- Model Assumptions:
	- While predicting POS in simplified if the word does not appear in the train set then it predicts noun for the corresponding word
	- While calcuating tau values in variable elimination, when the length of a sentence is large, mulyiplying subsequent tau values
	results in 0. To counter that we have introduced a factor of  10^(-20) if tau gets 0
	- While calculating posteriors, we have introduced a factor of 10^(-20) to counter 0 value when we take the log
	-While calculating the transition and emission probabilities for the words that are enconutered in testing set for the first time have been asssigned the lowest
	 possible probability value by assuming the numerator of likelihood to be less than that of 1.

- Model Accuracy for each algorithm
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
     1. Simplified:       93.92%               47.50%
         2. HMM VE:       94.65%               51.45%
        3. HMM MAP:       94.71%               51.90%

'''
####

import random
import math
import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:    
    def __init__(self):
       	self.pos_list_dict = {} #saves the occurences of pos tags {noun : 5 , verb : 6...}
        self.word_list_dict = {} #saves all the word occurences to respective pos tags (integrated data struct) ex {'noun' :{'word1': 2, 'word2' : 3...}, 'verb':{'word2' : 5...}...} 
        self.word_count_dict = {}# word occurences
        self.transition_count_dict = {} #saves all occurences of transition states {(noun, verb) : 5,.... }
        self.transition_prob_dict = {}
        self.starting_word_pos = {} #{noun in start: 5,.... }
        self.starting_prob_pos = {} # starting probabilities of pos tags
    	self.emission_probs_dict = {}
    	self.word_list_length = {}
    	self.emission_probs_dict = {}
    	self.alpha = 10**(-20)
        self.totalTags = 0
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label): 
        emissions = 1
        transitions = 1
        for i in range(len(sentence)):
            emissions *= self.getEmission(sentence[i], label[i])
        for i in range(len(sentence)-1):
            transitions *= self.getTransitionProb(label[i], label[i+1])

        return math.log(emissions*transitions + self.alpha)
    
    def getEmission(self, word, tag):
        emission = (word, tag)
        if(self.emission_probs_dict.has_key(emission)):
            return self.emission_probs_dict[emission]
        else:
            tag_for_word = self.word_list_dict[tag][word] if self.word_list_dict[tag].has_key(word) else 0.000001
            tag_length = self.pos_list_dict[tag] + len(self.word_count_dict)
            emission_prob = tag_for_word / float(tag_length) 
            self.emission_probs_dict[emission] = emission_prob
        return emission_prob
        
    def addWordToPos(self, word, wordtag):
        if(self.word_list_dict[wordtag].has_key(word) == False):
            self.word_list_dict[wordtag][word] = 1
        else:
            self.word_list_dict[wordtag][word] += 1
            
    def addTransitionCount(self, first_state, next_state):
    	if(self.transition_count_dict[first_state].has_key(next_state) == False):
    		self.transition_count_dict[first_state][next_state] = 1
    	else:
    		self.transition_count_dict[first_state][next_state] += 1
    		
    def getTransitionProb(self, prev_state, curr_state):
	prev_to_curr = (prev_state, curr_state)
	if(self.transition_prob_dict.has_key(prev_to_curr)== False):
	    	prev_to_next_count = self.transition_count_dict[prev_state][curr_state] if self.transition_count_dict[prev_state].has_key(curr_state) else 0.0001
	    	total_count_prev = sum(self.transition_count_dict[prev_state].values()) + len(self.pos_list_dict)
	    	prev_to_curr_prob = prev_to_next_count / float(total_count_prev)
	else:
		return self.transition_prob_dict[prev_to_curr]
	return prev_to_curr_prob
	
    # Do the training!
    #
    def train(self, data):
        for i in data:
            for x in range(0, len(i[0])):
                word = i[0][x]	
                wordtag = i[1][x]
                if(self.pos_list_dict.has_key(wordtag) == False): 
                    self.pos_list_dict[wordtag] = 1
                    self.word_list_dict[wordtag] = {}
                    self.addWordToPos(word, wordtag)
                else:
                    self.pos_list_dict[wordtag] += 1
                    self.addWordToPos(word, wordtag)
                if(self.word_count_dict.has_key(word) == False):
                	self.word_count_dict[word] = 1
                else:
                	self.word_count_dict[word] += 1
                if(x < len(i[0])-1):
		        next_tag = i[1][x+1]
			if(self.transition_count_dict.has_key(wordtag) == False): 
			        self.transition_count_dict[wordtag] = {}
			        self.addTransitionCount(wordtag, next_tag)
			else:
				self.addTransitionCount(wordtag, next_tag)
                if(x == 0):
                    if(self.starting_word_pos.has_key(wordtag) == False):
                        self.starting_word_pos[wordtag] = 1
                    else:
                        self.starting_word_pos[wordtag] += 1
                self.totalTags += 1

        self.initial_prob_dict = dict.fromkeys(self.pos_list_dict, 0)
        for key in self.initial_prob_dict:
            self.initial_prob_dict[key] = self.pos_list_dict[key] / float(self.totalTags)

        for key in self.starting_word_pos:
            starting_prob = self.starting_word_pos[key] / float(sum(self.starting_word_pos.values()))
            self.starting_prob_pos[key] = starting_prob
            
        for state, data in self.transition_count_dict.items():
        	for next_state in data:
        		transition = (state, next_state)
        		self.transition_prob_dict[transition] = self.getTransitionProb(state, next_state)

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        seq = []
        memoize = {}
        for word in sentence:
            self.probs_dict = dict.fromkeys(self.pos_list_dict, 0)
            for tag in self.probs_dict:
                if (self.word_list_length.has_key(tag) == False):
                    tag_length = sum(self.word_list_dict[tag].values())
                    self.word_list_length[tag] = tag_length
                else:
                    tag_length = self.word_list_length[tag]
                
                init_pro = self.initial_prob_dict[tag]
                tag_for_word = self.word_list_dict[tag][word] if self.word_list_dict[tag].has_key(word) else 0.1
                
                word_tag_tup = (word, tag)
                if(memoize.has_key(word_tag_tup) == False):
                    memoize[word_tag_tup] = 0 
                    prob_word_tag = init_pro * tag_for_word / float(tag_length)
                    memoize[word_tag_tup] = self.probs_dict[tag] = prob_word_tag 
                else:
                    self.probs_dict[tag] = memoize[word_tag_tup]
            if max(self.probs_dict.values()) != 0:
                    seq.append([key for key,value in self.probs_dict.items() if value == max(self.probs_dict.values())][0])
            else:
                    seq += ['noun']
        return seq
    
    def cal_tau(self, sentence):
        tau = {i: 0 for i in range(len(sentence))}
        for j in tau.keys():
            tau[j] = {i: 0 for i in self.pos_list_dict}
        for i, word in enumerate(sentence):
            for state in self.pos_list_dict:
                if i == 0:
                    tau[i][state] = self.pos_list_dict[state] / float(sum(self.pos_list_dict.values()))
                else:
                    tau[i][state] = sum(tau[i-1][k] * self.getTransitionProb(k, state) * self.getEmission(sentence[i-1] , k) for k in self.pos_list_dict)
                    tau[i][state] = self.alpha if tau[i][state] == 0 else tau[i][state]
        return tau

    def hmm_ve(self, sentence): 
        seq = []
        fwd_prb = {i: 0 for i in range(len(sentence))}
        for j in fwd_prb.keys():
            fwd_prb[j] = {i: 0 for i in self.pos_list_dict}       
        bwd_prb = copy.deepcopy(fwd_prb)
        tot_prb = copy.deepcopy(fwd_prb)        
        tau1 = self.cal_tau(sentence)
        for i, j in enumerate(sentence):
            for s in self.pos_list_dict:
                fwd_prb[i][s] = tau1[i][s] * self.getEmission(j , s)
        for i, j in enumerate(reversed(sentence)):
            for s in self.pos_list_dict:
                if i == 0:
                    bwd_prb[len(sentence)-i-1][s] = 1
                else:
                    bwd_prb[len(sentence)-i-1][s] = sum(self.getTransitionProb(s, k) * self.getEmission(sentence[len(sentence)-i] , k) for k in self.pos_list_dict)
        for i in range(len(sentence)):
            for s in self.pos_list_dict:
                tot_prb[i][s] = fwd_prb[i][s] * bwd_prb[i][s]        
        for i in tot_prb.keys():
            seq.append(max(tot_prb[i], key = tot_prb[i].get))
        return seq

    def hmm_viterbi(self, sentence):
        seq = []
        self.viterbi_lookup = []
        self.viterbi_probs = {}
        self.probs_dict_viterbi = {}
        tag_list = [tag for tag in self.pos_list_dict]
        states = [tag for tag in self.pos_list_dict]
        for counter, word in enumerate(sentence):
        	self.viterbi_lookup.append({})
        	for state in tag_list:
        		emission_prob = self.getEmission(word, state)
        		if counter == 0:
        			self.viterbi_lookup[counter][state] = {'prob' : emission_prob * self.starting_prob_pos[state] if self.starting_prob_pos.has_key(state) else 0.01, 'prev_state': None}
        		else:
        			max_val_state = self.getMaxValueViterbi(state, counter)
        			max_value, max_state = max_val_state[0], max_val_state[1] 
        			self.viterbi_lookup[counter][state] = {'prob' : emission_prob * max_value, 'prev_state' : max_state}
        max_prob = max(value['prob'] for value in self.viterbi_lookup[-1].values()) # get the max value for the last state
        previous = None
        for state, data in self.viterbi_lookup[-1].items():
        	if data["prob"] == max_prob:
             		seq.append(state)
		        previous = state
        		break
        		
	for counter in range(len(self.viterbi_lookup) - 2, -1, -1):
        	seq.insert(0, self.viterbi_lookup[counter + 1][previous]["prev_state"])
        	previous = self.viterbi_lookup[counter + 1][previous]["prev_state"]
        return seq
        
    def getViterbiSeq(self, seq, counter):
    	if counter == (len(self.viterbi_lookup)-1):
    		max_prob = max(value['prob'] for value in self.viterbi_lookup[-1].values())
   
    def getMaxValueViterbi(self, state_j, counter):
        tempDict = {}
        pos_tags = [i for i in self.pos_list_dict]
        for state_i in pos_tags:
            prev_state_prob = self.viterbi_lookup[counter - 1][state_i]['prob'] 
            transition_prob = self.getTransitionProb(state_i, state_j)
            tempDict[state_i] = prev_state_prob * transition_prob
        return max(zip(tempDict.values(), tempDict.keys()))
        
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

