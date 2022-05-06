# Text analysis

To better test our hypothesis of the forums becoming 'echo chambers' for the same opinions, we wanted to look at two things in particular: How much 'the same things' are said across each subreddit, as well as about the discussion in general. We also wanted to analyze the general sentiment of across the subreddits, to see if there was any indication that particularly negative or positive content was being propagated, adding to the 'echo chamber' factor.


## Analyzing sentiment scores

To get the sentiment score, we use VADER (https://www.researchgate.net/publication/275828927_VADER_A_Parsimonious_Rule-based_Model_for_Sentiment_Analysis_of_Social_Media_Text). The strength of VADER being that it is fairly resistant to things that confuse other algorithms for the same purpose; typically sentiment analysis algorithms work pretty straightforwardly, adding postive sentiment for each positive word in a text, vice versa for negative. The obvious problem here, being with more complex sequences of words, such as those that contain negations or strength indicators, for example; 'not good' and 'extremely bad' being two examples of such. VADER fixes this by considering not only single words at a time, but a combination of different n-grams, sequences of number of words, this allows it far more consideration of context than other approaches.

With this, we hope to find if any subreddits stand out particularly if they diverge fra from neutral, 0 sentiment score, meaning they are either overly positive or overly negative. This could particularly be important in the debate of internet forums becoming 'echo chambers' for negative emotions, more than places for actual discourse.

(We would also have liked to use this for comparing sentiment score to comment score, but due to the unreliability of upvotes/downvotes, we did not go through with this idea). The results from the sentiment analysis can be seen below.

1. television': 0.04185013090808849,
1.  'willsmith': 0.020197949036668706,
1.  'entertainment': 0.00014266911807619216,
1.  'Oscars': 0.021183817204300976,
1.  'OutOfTheLoop': 0.050206621556307326,
1.  'boxoffice': 0.03953054695562434,
1.  'news': 0.02232362648029532,
1. 'worldnews': 0.022202027027027012,
1. 'offbeat': -0.03620582524271847,
1.  'USNEWS': -0.12266666666666665

What is most interesting here is probably ht e outlier that is USNEWS... The VADER implementation generally advises, that one a sentiment score between -0.05 and 0.05 should be considered as neutral, by this logic, only the USNEWS and just barely OutOfTheLoop, come by as negative and postiive respectively. USNEWS might be explainable - the immediate reaction by the hollywoord and other larger actors guild organisations (organisations that one would think USNEWS would focus on), was to denounce Will Smith as the aggressor. This is expressed in examples such as Will Smith being barred from the Oscars the next 20 years.

Of course the body of a submission does not wholly explain the body of each comment, but it lends itself somewhat to intuition, that the comments of a submission be more likely to follow the sentiment of the submission, than not. Analyzing in depth whether or not comments follow the submission in sentiment would be interesting in a future study.


## Wordcloud and Joke Context

To get an idea about the debate about the slap, we divide the comments into documents and create wordclouds based on the TF-IDF scores. This is because wordclouds are a very visual way of summarizing a big corpora of documents. Hence, it allows us to combine the computational power of a computer with the human power of pattern recognition and logical deduction. The process is the following. First we decide what category to group them by. For example, let’s work with subreddit, such that each subreddit represents one document. Then to create the documents, we loop over all the posts (i.e. submissions and comments), and we tokenize, stem and remove stopwords. Then for each unique word that is left in the post, we add that word to the document with a score of one. This means that when we’re done iterating over the entire dataset, for each document each word in that document has a word count that represents the number of posts from that category that contains that word - not the number of appearances of that word in total. The reason why we do this is to downweigh the long posts. Let’s imagine what would happen if we didn’t. Then words that are important in long posts would have a much higher score than the important words in shorter posts. This would bias our analysis towards the long comments, and we wan’t to capture the variety of the debate. 
Now we can calculate the TF-IDF scores for each word in each document, and use that to plot the wordclouds. The TF-IDF score for the word t in document d is defined as follows:

$$TF-IDF(t,d) = \frac{f_{t,d}}/{\sum_{t\in d} f_t,d} \cdot \log \frac {N}{|\{d\in D|t\in d\}|}$$

Here, D is the set of documents, i.e the corpora, and f_{t,d} is the raw count of the word t in document d. In the practical implementation in this exercise however, we don’t divide by the total number of words in the document, $$\sum_{t\in d} f_t,d$$. This is because we only use the TF-IDF scores to compare the words within documents, so that operation just scales the scores, and doesn’t change the relations between them. 
With that said, let’s look at some word clouds. First we will examine the division we just talked about, where each document represents a subreddit, and we will only examine the 9 largest. 

![](/images/wordcloud_subreddit.png)
Here we see that unsurprisingly one word that apears very often is ‘joke’. Therefore, let’s take a look at some of the context of the word ‘joke’ in the dataset.
![](/images/wordcloud_topic.png)

Even here we see that some of the comments actually defend Will for his actions. One example is the last one that says “it’s possible to try to laugh off a joke and realize you’re actually hurt (...)”. Some people also say that the joke was tasteless. However, most of them seem to defend Chris Rock even though they think the joke is tasteless
Furthermore, we can see that even though we’re only looking at posts are reposts to comments about either Will Smith, his wife, Chris Rock, the slap and the Oscars, many other words come into play in some of the subreddits, e.g. worldnews, boxoffice and offbeat. This is interesting as it shows how unfocused a discussion can be, and how fast the attention of humas can wander off - especially on the internet. Another interesting thing is the willsmith subreddit. Here, none of the dominant words from the other subreddits are not nearly as dominant. This indicates that on the willsmith subreddit they didn’t really care that much about the slap. Admittedly, ‘joke’, ‘wife’, ‘jada’ etc appear here as will, but many other words are just as important.
Let’s move on to a wordcloud plot where we have divided the words according to topic, that is each of the elements in [Chris Rock, Will Smith, Slap, Oscar, Jada Pinkett] represent a document. Note that here, each post can mention multiple topics, so the same post can be accounted in multiple documents.
[WORDCLOUD]
Here we see a much bigger variation, and also, many of the words seem a bit random. This is because we only have 5 documents in this case, and looking at the TF-IDF equation, if a word appears in all the documents, the TF-IDF score becomes zero. However, it is interesting, that words like ‘polygamous’, ‘monogamy’ and ‘soulmate’ appear frequently in the topic of jada smith. This is interesting because it aligns with the general tendency to assign these sort of “soft” values to the women.
Now let’s move on to the graph analysis


## Analyzing syntatic simliarity

Another tool we used to analyze the general similarity of the text within subreddits was the syntactic similarity of each comment in a subreddit when considering the other comments in that subreddit, here, we use the Facebook-developed algorithm: Fastext. This allows us to vectorize each submission text to obtain a vector with 100 features that describe the abstract syntatic content of a given submission. In practice, these vectors are created by considering submissions that contain many words that appear in the same contexts as syntactically similar, specifically, it uses a skipgram model to obtain vector embeddings.

After creating the vectors for each submission, we can compare the syntactic similarity of each submission in a given subreddit to each other submission in the same subreddit by simply comparing the similarity of their respective vectors. Finally, we can weigh these simliarities together to obtain an average 'simliarity score' for each subreddit. The actual vector similarities for each pair of two vectors, were calculated by the cosine simliarity:

$$sim(v_1, v_2) = \frac{v_1 \bullet v_2}{||v_1|| \times ||v_2||}$$

However, in practice, since there are so many unique submissions, it is completely infeasible to compare each comment in each subreddit to each other comment, as this would lead to hundreds of billions of comparisons. Instead we sample randomly from the pool of submissions from each subreddit, and calculate the simliarity score between these samples instead. With this, we assume that the average simliarity score of our samples are somewhat representative of the overall simliarity score of their respective subreddits.

The results for the syntatic simliarity test were as follows:

1. 'Oscars': 0.5734081361840402
1. 'OutOfTheLoop', 0.6267408389493618
1. 'USNEWS', 0.7517130391465293
1. 'boxoffice', 0.6556357684156545
1. 'entertainment', 0.5915876663660113
1. 'news', 0.5885688143422156
1. 'offbeat', 0.6371274988840921
1. 'television', 0.6832170959722129
1. 'willsmith', 0.6446319649818191
1. 'worldnews', 0.62317395577479

As it can be seen, the greatest amount of syntactc simliarity is found in USNEWS - most likely explainable as most content there are of the same type - news articles. Another interesting point, however, is how generally high this syntactic similarity score is. 

