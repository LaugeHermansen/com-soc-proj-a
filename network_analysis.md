# Methods for graph analysis

We wanted to analyze how each comment and submission interacts with one another, and the most optimal way, we decided, was to create a graph where the nodes are unique users and the edges their interactions with other comments or submissions. In all cases, these are weighted, but we consider them either directed or undirected. In the undirected case, the weight is simply the sum of replies one user has made to another user's submissions (comments or posts) and vice versa, in the directed case we simply keep track of these two values independently.

## The degree distribution of the graph

First, we wanted to find out if the graph we had created modelled something natural, and not just random interactions between random users. To do this, we looked at the degree distribution of the created graph.

Looking at the degree distribution of the graph made by linking redditors by comments and submissions, we see that there is a clear tendency towards most nodes having an out degree of 0, as seen by the plot. The distribution of nodes seems to follow a logarithmic trend, this is especially evident when looking at the in-degree distribution of the directed graph. Here, very few nodes have an extremely high in-degree, most likely indicating those who have posted either very popular comments or are submission 'owners' of popular submissions.

![](/images/non_random.png)

We compare this to a random distribution created by iterating through each node and adding edges to each other node with probability p, where p is given by:

$$p = \frac{k_{\mu}}{N-1}$$

Where $k_{\mu}$ is the average degree of a given node in the network. Our hypothesis is that the degree distribution of a random network should be quite different than that of the redditor network

![](/images/random.png)

We see that the two graphs look quite dissimlar, implying that the graph of redditors indeed is a natural graph, rather than a purely random one.


## Analyzing modularity in the graph

As we want to examine how much people branch out from just one subreddit, we use modularity to examine this. Modularity is a measure of how well a given community 'fits' a graph. That is, given a graph and a unique set of nodes for each possible community, the modularity weighs how likely a node in any community is to branch out to other communities, weighted by how likely they are to stay within their own community. A modularity close to 0 implies that there is no inherent community structure, a value close to 1 implies that the modularity structure given is most likely a true indication of the communities, while one close to -1 implies the wrong communities. We group each user in a community based on which subreddit they have made most submissions to, and then calculate the modularity of the resulting graph. As a baseline, we use Louvain's algorithm which optimizes communities in the graph, so as to produce the highest possible modularity. Note, that while we only work with 11 communities in the case of subreddits, Louvain's algorithm can create as many number of communities as will needed to maximize the modularity.

The values for modularity were **0.581** and **0.784** for assigning users to community based on their most popular subreddit and by Louvain's algorithm. Both of these values are quite high. Louvain's algorithm is of course expected to give a high value in any graph were any type of community, explicit or otherwise exists. The high value of modularity when simply assigning by popularity is interesting, however. This could imply, that redditors in this case, primarily stick to one or a few subreddits, rather than branch out to many.


