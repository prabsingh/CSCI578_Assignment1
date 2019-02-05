import numpy as np
import scipy.stats as sp

vocab_size = 50
alpha = [.8] * vocab_size
topic_size = 5
doc_size = 50
num_docs = 10
doc_array = [None]* (topic_size *num_docs)
mu_array = [None] * topic_size

def gen_topics():
    for i in range(0, topic_size):
        mu_array[i] = np.random.dirichlet(alpha)

def gen_docs():
    for topic in range(0, topic_size):
        for i in range(0, num_docs):
            doc_array[topic * num_docs + i] = np.random.multinomial(doc_size, mu_array[topic])
def gen_docs_mixed():
    B_array = [0] * topic_size
    B_array[0] = [.3, .3, .1, .1, .2]
    B_array[1] = [.4, .1, .5, 0, 0]
    B_array[2] = [.1, .2, 0, 0, .7]
    B_array[3] = [.1, .1, .5, .3, 0]
    B_array[4] = [0, .3, 0, .5, .2]

    #for each class
    for i in range(0, topic_size):
        #create 10 docs
        for j in range(0, num_docs):
            #How many words are from each mu
            words_from_mu = np.random.multinomial(doc_size, B_array[i])
            doc = [0] * vocab_size
            #for each mu, sample the corrresponding number of words
            #add the counts to the doc
            for k in range(0, 5):
                 values = np.random.multinomial(words_from_mu[k], mu_array[k])
                 for word in range(0, vocab_size):
                     doc[word] += values[word]
            doc_array[i * num_docs + j] = doc
                 
            

def create_histogram():
    histogram_counts = np.zeros((topic_size, vocab_size), dtype=float)
    for i in range(0, topic_size * num_docs):
        for j in range(0, vocab_size):
            histogram_counts[i//num_docs][j] += doc_array[i][j]
    #graph the histogram
            
def calc_mu_ML(data_array):
    mu = np.zeros(vocab_size, dtype=float)

    if(len(data_array) == 0):
        return mu
    
    for doc in data_array:
        for word in range(0, vocab_size):
            mu[word] += doc[word]

    total_samples = len(data_array) * doc_size

    for word in range(0, vocab_size):
        mu[word] /= total_samples

    return mu

#P(T|D) = P(D|T)P(T)/P(D)
#P(T|D) ~ P(D|T)P(T)
#P(T|D) ~ P(D|T) Count(T)
def naiveBayes(doc_array):
    random_order = np.random.permutation(50)

    train = random_order[:25]
    test = random_order[25:]

    count_T = [0] * 5
    for i in train:
        count_T[i//10] +=1

    #calculate mu_array of ML 
    mu_array = [0] * topic_size
    for i in range(0, topic_size):
        topic_doc_array = []
        for item in train:
            if (item//10) == i:
                topic_doc_array.append(doc_array[i])
        mu_array[i] = calc_mu_ML(topic_doc_array)

    correct_predict_count = 0
    
    for doc_num in test:
        current_large = 0
        current_class = 0
        # find the topic with the greatest prob of generating the data
        for i in range(0, topic_size):
            likelyhood = sp.multinomial.pmf(doc_array[doc_num], doc_size, mu_array[i])
            prior = count_T[i]
            if(likelyhood * prior > current_large):
                current_large = likelyhood * prior
                current_class = i

        #compare the guess with actual
        if current_class == doc_num//10:
            correct_predict_count += 1
    return correct_predict_count/25
 
def main():
    #part 4a
    gen_topics()

    #part 4b
    gen_docs()

    #part 4c
    create_histogram()
    mu_ML = [0] * topic_size
    for i in range(0, topic_size):
        mu_ML[i] = calc_mu_ML(doc_array[i * num_docs :(i + 1) * num_docs])

    #part 4d
    accuracy = naiveBayes(doc_array)

    #part 4e
    accuracy_array = [0] * 100  
    for i in range(0, 100):
        gen_topics()
        gen_docs()
        accuracy_array[i] = naiveBayes(doc_array)
    mean = np.mean(accuracy_array)
    sd = np.std(accuracy_array)
    print(mean)
    print(sd)
    #part 5
    gen_topics()
    gen_docs_mixed()
    accuracy_mixed = naiveBayes(doc_array)

    accuracy_array_mixed = [0] * 100
    for i in range(0, 100):
        gen_topics()
        gen_docs_mixed()
        accuracy_array_mixed[i] = naiveBayes(doc_array)
    mean_mixed = np.mean(accuracy_array_mixed)
    sd_mixed = np.std(accuracy_array_mixed)

    print(mean_mixed)
    print(sd_mixed)

main()
