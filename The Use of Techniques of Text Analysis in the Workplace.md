# The Use of Techniques of Text Analysis in the Workplace

## Background
This blog reflects my thought after completing [this project](). This project is about utilizing text analysis techniques to analyze unstructured data (text) in 42 text documents, aiming at providing insights and figuring out hidden themes in these documents.

The steps of text analysis I performed are summarized below to show the foundation of later reflection:
1.	**Data preparation** – Remove punctuation, stop words and numbers, stemming etc.
2.	**Exploratory data analysis (EDA)** – Analysis and visualization of word frequency count, bigram and TF-IDF etc.
3.	**Clustering** – Various methods including K-mean, Hierarchical, Fast Greedy and Louvain clustering.
4.	**Latent semantic analysis (LSA)** – Analysis of document similarity and finding out the nearest neighbors of specific words.
5.	**Topic Modelling** – Find hidden topics and corresponding documents among all 42 documents.
6.	**Sentiment Analysis** – Overall sentiment analysis for each document to provide a basic understanding of sentiment.

The techniques I used in the assignment can be classified as follows:

1.	Text classification: Sentiment analysis, Topic modeling, LSA
2.	Word Frequency:  EDA
3.	Clustering: clustering

## Reflection

Text analysis techniques would be extremely helpful to organizations that need to deal with a huge amount of information like email, chats, social media, documents and so on. Analyzing a vast amount of text documents is tedious, and time-consuming and the interpretation might not be consistent. However, a well-trained text analysis algorithm/ model can work 24-7 and output consistent results. Organizations can redirect human resources to other human-required tasks by implementing text analysis techniques.

### Text classification

#### Sentiment analysis
There are many ways customer can communicate their thoughts and request about products and services, including but not limited to surveys and email. Companies using sentiment analysis can easily find out complaints and urgent requests as the techniques can output a score indicating either an inquiry is positive or negative, the extent is shown on the value of the score. So they can process according to the urgency to reduce customer unsatisfactory.

#### Topic Modelling
Topic modeling divide text (data) into separate groups by theme. It is useful when an organization analysis its customer group and the feedback. For example, a company can group feedback by different topics, say there is a topic relating to delivery. The data that fall into this topic can be pulled out for further analysis. It may show that customers in one area are suffering from delayed problem and customer in another area has no complaints at all. This is just an example showing how topic modeling could reveal the potential problem-solving path.

#### Word Frequency
Word frequency (including a single word, bigram and trigram) is a technique showing the words with the most occurrence, which could be a good indicator to companies what major questions or issues are concerning among a vast number of customers. TF_IDF is another similar method giving a weight measurement of each word, providing another view of the most concerning issue. For example, if a company is analyzing a survey result of their product, both word frequency and TF_IDF can generate a word cloud clearly showing the words with the most occurrence and TF_IDF weight. The management team can evaluate the word say delivery, function etc. and look for a solution.

#### Clustering
Clustering is like topic modeling, basic is to group a large amount of unstructured data by their distance. There are numerous algorithms to cluster, and a few ways to map the distance between documents, selection of the algorithm should be considered thoroughly as different algorithm yield different results and accuracy depending on the scenario. Though the clustering algorithm is less accurate than the classification algorithm, it is much faster to implement and provides an opportunity for fuzzy search. A good example would be Google search, Google’s cluster algorithm groups websites into a cluster based on their similarity and that is why it takes no time to show websites relevant to what people searched for.

#### Combining Text Analysis Techniques
The abovementioned techniques are a breakdown of Text analysis just to show some examples, in the workplace, data scientist always utilizes all the appropriate technique he needs to output meaningful insights. It can not only be used to identify the inquiry’s urgency and find out the hidden problem, it can also be used broader. For example, companies can scrap data from the internet like Facebook and Twitter about customers’ feelings about their service and product and seek improvement or maintain existing customers. It can also be used in competitor research, to understand competitors better and seek opportunities to take advantage of them. Furthermore, companies can analyze their material supplier and/or manufacturer, and their competitors might be able to find better deals or better-quality material suppliers and/or manufacturers.

## Conclusion
To wrap up, text analysis helps deal with vast quantities of unstructured data and yield more consistent and accurate results compared to the human being. It improves efficiency and frees humans from repetitive work, the most valuable and used scenario is efficiently providing an accurate understanding of customer insights from a large amount of social media posts, online feedback, and survey responses.

