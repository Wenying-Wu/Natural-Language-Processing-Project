# Project The Anatomy of an Unknown Corpus                                 #####  
# By: Wenying Wu 
##-----------------------------------------------------------------------------#
# .------------- Set up -------------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# Set working directory ----
setwd("D:/Machine Learning/NLP") 
Sys.setlocale("LC_TIME", "English")

# Load packages ----
library(tm) 
library(SnowballC) 
library(cluster) 
library(wordcloud) 
library(ggplot2) 
library(dplyr) 
library(topicmodels) 
library(ape) 
library(kableExtra)
library(ggrepel)
library(sentimentr)
library(tidyverse)
library(tidytext)
library(ggcorrplot)
library(igraph)
library(visNetwork)
library(textdata)
library(lsa)
library(stringi)
library(LSAfun)

#library(srvyr)

# Read docs ----
docs <- VCorpus("DirSource"("./docs"))

# Print docs 
print(docs)
class(docs)
summary(docs)
# Volatile corpus consisting of 42 documents
# Examine contents, pick doc 01 as example
docs[1]
class(docs[[1]])
docs[[1]]$meta
docs[[4]]$content

# Pre-processing ----
# Setting up Stop Words
myStopwords <- c("can", "say","one","way","use",
                 "also","howev","tell","will",
                 "much","need","take","tend","even",
                 "like","particular","rather","said",
                 "get","well","make","ask","come","end",
                 "first","two","help","often","may",
                 "might","see","someth","thing",
                 "post","look","right","now","think","'ve ",
                 "'re ","anoth","put","set","new",
                 "want","sure","kind","larg","yes,","day","etc",
                 "quit","sinc","attempt","seen","awar",
                 "littl","ever","moreov","though","found","abl",
                 "enough","far","earli","away","draw",
                 "last","never","brief","bit","entir",
                 "lot","man","say")

# Remove punctuation - replace punctuation marks with " "
docs <- tm_map(docs, removePunctuation)
# Transform to lower case
docs <- tm_map(docs,content_transformer(tolower))
# Remove digits
docs <- tm_map(docs, removeNumbers)
# Stem to standardise variants
docs <- tm_map(docs,stemDocument)
# Remove stopwords from standard stopword list and own stopword list (myStopwords)
docs <- tm_map(docs, removeWords, c(stopwords("english"),myStopwords))
# Remove whitespaces
docs <- tm_map(docs, stripWhitespace)

# Inspect output of docs1
writeLines(as.character(docs[[4]]))

##-----------------------------------------------------------------------------#
# .------------- EDA ----------------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# Part A: Using word frequency ####
# Create document-term matrix
dtm <- DocumentTermMatrix(docs)
# Summary of the Document Term Matrix
dtm
#inspect segment of document term matrix
inspect(dtm[1:10,1000:1006])
## Collapse matrix by summing over columns 
# - this gets total counts (over all docs) for each term
freq <- colSums(as.matrix(dtm))
# Number of terms 
length(freq)
# Create sort order (asc)
ord <- order(freq,decreasing=TRUE)
# Inspect most frequently occurring terms
freq[head(ord)]
# List most frequent terms.
findFreqTerms(dtm,lowfreq=100)
Freqtable <- data.frame(word=names(freq), freq=freq)  

# a. Histogram ordered by frequency ----
wf=data.frame(term=names(freq),occurrences=freq)
subset(wf, occurrences>200) %>% 
  ggplot(aes(reorder(term,-occurrences), occurrences)) +
  geom_bar(stat="identity", colour="black") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(title = "Most Frequent Words",
       x ="Words",
       y = "Frequency")

# b. Generate Wordcloud ----
# Setting the same seed each time ensures consistent look across clouds
set.seed(42)
# Generate Wordcloud with min of 180
wordcloud(names(freq),freq,min.freq=180,colors=brewer.pal(8,"Dark2"),
          scale=c(3,0,8), random.order = FALSE, use.r.layout = TRUE)

# c. Create bigram Tokenizer using word frequency #### 
Tokenizer <-  function(x) unlist(lapply(ngrams(words(x), c(2)), paste, collapse = " "),
                                 use.names = FALSE)
# Create DTM 
dtmbi <- DocumentTermMatrix(docs, control = list(tokenize = Tokenizer))
freqbi <- colSums(as.matrix(dtmbi))
wof <- data.frame(word=names(freqbi), freq=freqbi)
# Total number of terms 
length(freqbi)
# Create sort order (asc)
ordbi <- order(freqbi,decreasing=TRUE)
# Inspect most frequently occurring terms
freqbi[head(ordbi,n=50)]
# Plot frequently occurring terms
freq <- sort(colSums(as.matrix(dtmbi)), decreasing=TRUE)
wof <- data.frame(word=names(freq), freq=freq)
subset(wof, freq > 30) %>% 
  ggplot(aes(reorder(word, -freq),freq)) +
  geom_bar(stat="identity", colour="black") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(title = "Most Frequent Bi-Gram",
       x ="Words",
       y = "Frequency")


# Part B: Using TfIdf ####
# TfIdf: Term frequency - inverse document frequency, 
# The TfIdf weighting is normalised by the number of terms in the document.
dtm_tfidf <- DocumentTermMatrix(docs,control = list(weighting = weightTfIdf))
# Summary of the Document Term Matrix
dtm_tfidf
# Collapse matrix by summing over columns - this gets total counts (over all docs) for each term
wt_tot_tfidf <- colSums(as.matrix(dtm_tfidf))
# Number of terms 
length(wt_tot_tfidf)
# Create sort order (asc)
ord_tfidf <- order(wt_tot_tfidf,decreasing=TRUE)
# Inspect most frequently occurring terms 
wt_tot_tfidf[head(ord_tfidf)]
# Inspect least frequently occurring terms 
wt_tot_tfidf[tail(ord_tfidf)]

# a. Histogram ordered by TfIdf weighting ----
wf=data.frame(term=names(wt_tot_tfidf),weights=wt_tot_tfidf)
subset(wf, wt_tot_tfidf>0.2) %>% 
  ggplot( aes(reorder(term,-weights), weights)) +
  geom_bar(stat="identity", colour="black") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(title = "Most TF-IDF Weighted Words",
       x ="Words",
       y ="Weights")
# b. Generate Wordcloud 
# Setting the same seed each time ensures consistent look across clouds
set.seed(42)
wordcloud(names(wt_tot_tfidf),wt_tot_tfidf,max.words=18,colors=brewer.pal(8,"Dark2"),
          scale=c(3,0.8), random.order = FALSE, use.r.layout = TRUE)


##-----------------------------------------------------------------------------#
# .------------- Clustering ---------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# Part A: Hierarchical clustering #### 
# a. Using frequency ----
# Convert DTM to matrix 
m<-as.matrix(dtm)
# Cosine distance measure
cosineSim <- function(x){ 
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))}
cs <- cosineSim(m)
cd <- 1-cs
# Run hierarchical clustering using cosine distance
groups_f <- hclust(cd,method="ward.D")
# Plot trees 
plot(groups_f, hang=-1)
# Cut into 6 subtrees
rect.hclust(groups_f,6, border = "red")
# Cut into 5 subtrees
rect.hclust(groups_f,5, border = "red")


# b. Using Tfldf ----
# Convert DTM to matrix 
m_tfidf<-as.matrix(dtm_tfidf)
# Cosine distance measure
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))}
cs <- cosineSim(m_tfidf)
cd <- 1-cs
# Run hierarchical clustering using cosine distance
groups_TF_IDF <- hclust(cd,method="ward.D")
# Plot trees 
plot(groups_TF_IDF, hang=-1)
# Cut into 6 subtrees
rect.hclust(groups_TF_IDF,6, border = "red")
# Cut into 5 subtrees
rect.hclust(groups_TF_IDF,5, border = "red")


# Part B: K-Means Clustering ----
# a. Using frequency ----
# Cosine distance measure
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))
}
cs <- cosineSim(m)
cd <- 1-cs
kfit <- kmeans(cd, 5, nstart=100)
clusplot(as.matrix(cd), kfit$cluster, color=T, shade=T, labels=2, lines=0)

# Print contents of kfit
print(kfit)
# Print cluster sizes
kfit$size
# Print clusters members
kfit$cluster
# Sum of squared distance between cluster centers 
kfit$betweenss
# Sum of squared distance within a cluster 
kfit$withinss

# Determine optimal number of clusters for cosine distance
wss <- 2:(length(docs)-1)
for (i in 2:(length(docs)-1)) wss[i] <- sum(kmeans(cd,centers=i,nstart=25)$withinss)
plot(2:(length(docs)-1), wss[2:(length(docs)-1)], type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares") 

# b. Using TfLdf and cosine distance ----
# Cosine distance measure
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))
}
cs <- cosineSim(m_tfidf)
cd <- 1-cs
kfit <- kmeans(cd, 5, nstart=100)
clusplot(as.matrix(cd), kfit$cluster, color=T, shade=T, labels=2, lines=0)

# Print contents of kfit
print(kfit)
# Print cluster sizes
kfit$size
# Print clusters members
kfit$cluster
# Sum of squared distance between cluster centers 
kfit$betweenss
# Sum of squared distance within a cluster 
kfit$withinss

# Determine optimal number of clusters for cosine distance
wss <- 2:(length(docs)-1)
for (i in 2:(length(docs)-1)) wss[i] <- sum(kmeans(cd,centers=i,nstart=25)$withinss)
plot(2:(length(docs)-1), wss[2:(length(docs)-1)], type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares") 




##-----------------------------------------------------------------------------#
# .------------- Network ------------------------------------------------- #####          
##-----------------------------------------------------------------------------#
# Map filenames to matrix row numbers
# these numbers will be used to reference files in the network graph
filekey <- cbind(1:length(docs),rownames(m))
# have a look at file
rownames(m) <- 1:length(docs)
# compute cosine similarity between document vectors
# converting to distance matrix sets diagonal elements to 0
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))
}
cs <- cosineSim(m)
# adjacency matrix: set entries below a certain threshold to 0.
# We choose half the magnitude of the largest element of the matrix as the cutoff. 
# This is an arbitrary choice
cs[cs < max(cs)/2] <- 0
cs <- round(cs,3)
# build a graph from the above matrix
# mode is undirected because similarity is a bidirectional relationship
g <- graph.adjacency(as.matrix(cs), weighted=T, mode = "undirected")

# Plot a graph
set.seed(42)
layout1 <- layout.fruchterman.reingold(g)
comm_fg <- fastgreedy.community(g)
comm_fg$membership
V(g)$color <- comm_fg$membership 
community_mapping <- cbind(as.data.frame(filekey, row.names = F),comm_fg$membership)
community_mapping
## plot ----
data <- toVisNetworkData(g)
data$nodes <- data$nodes %>% mutate(font.size = 50)
visNetwork(nodes = data$nodes, edges = data$edges, 
           height = "500px", width = "100%") %>%
  visNodes(size = 10)

#Community detection - Louvain ----
comm_lv <- cluster_louvain(g)
comm_lv$membership
V(g)$color <- comm_lv$membership
community_mapping <- cbind(community_mapping,comm_lv$membership)
community_mapping
## plot ----
data <- toVisNetworkData(g)
data$nodes <- data$nodes %>% mutate(font.size = 50)
visNetwork(nodes = data$nodes, edges = data$edges, 
           height = "500px", width = "100%") %>%
  visNodes(size = 10)
#very similar clustering


##-----------------------------------------------------------------------------#
# .------------- Latent semantic analysis -------------------------------- #####          
##-----------------------------------------------------------------------------#
# Part A: Using frequency ----

## Create term-document matrix (lsa expects a TDM rather than a DTM)
tdm <- TermDocumentMatrix(docs)
## summary
tdm
## inspect segment of document term matrix
inspect(tdm[1000:1006,1:10])
## what kind of object is the tdm
class(tdm)
## convert to regular matrix
tdm.matrix <- as.matrix(tdm)
## check class
class(tdm.matrix)
dim(tdm.matrix)
## weight terms and docs
##  use the equivalent of tf-idf (local weight - tf, global -idf)
tdm.matrix.lsa <- lw_tf(tdm.matrix) * gw_idf(tdm.matrix)
dim(tdm.matrix.lsa)
## compute the Latent semantic space
lsaSpace <- lsa(tdm.matrix.lsa, dimcalc_share()) # create LSA space
## examine output
names(lsaSpace)

## .---Similarity ----
## Calculate similarity of documents in LSA space
LSAMat <- as.textmatrix(lsaSpace)
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))}
cs.lsa <- as.matrix(cosineSim(t(LSAMat)))
cormat <- reshape2::melt(cs.lsa)
pcorr=ggplot(data = cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+scale_fill_gradient2(low="red", mid="white", high="darkred")+
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(x =" ",
       y = " ",
       fill = "Similarity")
reorder_cormat <- function(cormat){
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]}
cs.lsa1=reorder_cormat(cs.lsa)
cormat1 <- reshape2::melt(cs.lsa1)
pcorr %+% cormat1

## .--- nearest neighbours ----
library(LSAfun)
dim(lsaSpace$tk)
LSAtk <- t(lsaSpace$sk*t(lsaSpace$tk))
neighbors("project",n=6,tvectors=LSAtk)
neighbors("risk",n=6,tvectors=LSAtk)
neighbors("manag",n=6,tvectors=LSAtk)
neighbors("figur",n=6,tvectors=LSAtk)
neighbors("time",n=6,tvectors=LSAtk)
neighbors("task",n=6,tvectors=LSAtk)

# Part B: Using TfIdf ####
## Create term-document matrix (lsa expects a TDM rather than a DTM)
tdm_tfidf <- TermDocumentMatrix(docs,control = list(weighting = weightTfIdf))
## summary
tdm_tfidf
## inspect segment of document term matrix
inspect(tdm_tfidf[1000:1006,1:10])
## what kind of object is the tdm_tfidf
class(tdm_tfidf)
## convert to regular matrix
tdm_tfidf.matrix <- as.matrix(tdm_tfidf)
## check class
class(tdm_tfidf.matrix)
dim(tdm_tfidf.matrix)
## weight terms and docs
## use the equivalent of tf-idf (local weight - tf, global -idf)
tdm_tfidf.matrix.lsa <- lw_tf(tdm_tfidf.matrix) * gw_idf(tdm_tfidf.matrix)
dim(tdm_tfidf.matrix.lsa)
## compute the Latent semantic space
lsaSpace <- lsa(tdm_tfidf.matrix.lsa, dimcalc_share()) # create LSA space
## examine output
names(lsaSpace)

## .---Similarity ----
## Calculate similarity of documents in LSA space
LSAMat <- as.textmatrix(lsaSpace)
cosineSim <- function(x){
  as.dist(x%*%t(x)/(sqrt(rowSums(x^2) %*% t(rowSums(x^2)))))}
cs.lsa <- as.matrix(cosineSim(t(LSAMat)))
cormat <- reshape2::melt(cs.lsa)
pcorr=ggplot(data = cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+scale_fill_gradient2(low="red", mid="white", high="darkred")+
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(x =" ",
       y = " ",
       fill = "Similarity")
reorder_cormat <- function(cormat){
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]}
cs.lsa1=reorder_cormat(cs.lsa)
cormat1 <- reshape2::melt(cs.lsa1)
pcorr %+% cormat1

## .--- nearest neighbours ----
dim(lsaSpace$tk)
LSAtk <- t(lsaSpace$sk*t(lsaSpace$tk))
neighbors("project",n=6,tvectors=LSAtk)
neighbors("risk",n=6,tvectors=LSAtk)
neighbors("manag",n=6,tvectors=LSAtk)
neighbors("figur",n=6,tvectors=LSAtk)
neighbors("time",n=6,tvectors=LSAtk)
neighbors("task",n=6,tvectors=LSAtk)

##-----------------------------------------------------------------------------#
# .------------- Topic Modeling ------------------------------------------ #####          
##-----------------------------------------------------------------------------#
# K = 6 ----
# Start from a representative point
burnin <- 1000
# Perform 2000 iterations 
iter <- 2000
# Use Thinning to ensure that samples are not correlated.
thin <- 500
# 5 different, randomly chosen starting points
nstart <- 5
# Using seeds. 
seed <- list(1,2,3,4,5)
# Take the run with the highest probability as the result
best <- TRUE
#Number of topics 
k <- 6

ldaOut <- LDA(dtm,k, method="Gibbs", control=
                list(nstart=nstart, seed = seed, best=best, 
                     burnin = burnin, iter = iter, thin=thin))
topics(ldaOut)
ldaOut.topics <-as.matrix(topics(ldaOut))
topic_sum <- data.frame(Topic = ldaOut.topics) 
topic_sum %>% 
  mutate(docs_name = stri_sub(rownames(topic_sum), 1, 5)) %>% 
  group_by(Topic) %>% 
  summarise(Docs = paste(docs_name, collapse = " "))

terms(ldaOut,8)
ldaOut.terms <- as.matrix(terms(ldaOut,8))

# Find probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma) 

# K = 5 ----
# Start from a representative point
burnin <- 1000
# Perform 2000 iterations 
iter <- 2000
# Use Thinning to ensure that samples are not correlated.
thin <- 500
# 5 different, randomly chosen starting points
nstart <- 5
# Using seeds. 
seed <- list(1,2,3,4,5)
# Take the run with the highest probability as the result
best <- TRUE
#Number of topics 
k <- 5

ldaOut <- LDA(dtm,k, method="Gibbs", control=
                list(nstart=nstart, seed = seed, best=best, 
                     burnin = burnin, iter = iter, thin=thin))
topics(ldaOut)
ldaOut.topics <-as.matrix(topics(ldaOut))
topic_sum <- data.frame(Topic = ldaOut.topics)
topic_sum$docs_name = stri_sub(rownames(topic_sum), 1, 5)

topic_sum %>% 
  dplyr::group_by(Topic) %>% 
  dplyr::summarise(Docs = paste(docs_name, collapse = " "))

terms(ldaOut,8)
ldaOut.terms <- as.matrix(terms(ldaOut,8))

# Find probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma) 



##-----------------------------------------------------------------------------#
# .------------- Sentiments ---------------------------------------------- #####          
##-----------------------------------------------------------------------------#

# Top words contributing to both positive and negative sentiments -----
docs_name  <- paste0("Doc", sprintf("%02d", c(1:42)))
series <- tibble()
for(i in seq_along(docs_name)) {
  clean <- tibble(text = docs[[i]]$content) %>%
    unnest_tokens(word, text) %>%
    mutate(docs_name = docs_name[i]) %>%
    select(docs_name, everything())
  series <- rbind(series, clean)
}
# set factor to keep books in order of publication
series$docs_name <- factor(series$docs_name, levels = rev(docs_name))

afinn <- series %>%
  group_by(docs_name) %>% 
  mutate(word_count = 1:n(),
         index = word_count %/% 500 + 1) %>% 
  inner_join(get_sentiments("afinn")) %>%
  group_by(docs_name) %>%
  dplyr::summarise(sentiment = sum(value)) %>%
  mutate(method = "AFINN") 

series %>%
  right_join(get_sentiments("nrc")) %>%
  filter(!is.na(sentiment)) %>%
  dplyr::count(sentiment, sort = TRUE)
afinn <- merge(afinn, topic_sum, by ="docs_name", all.x = TRUE) 
afinn$Topic <- as.factor(afinn$Topic)
afinn %>%
  ggplot(aes(reorder(docs_name, -sentiment), sentiment, fill = Topic)) +
  geom_bar(alpha = 0.8, stat = "identity")+
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  labs(title = "Sentiment Analysis",
       x ="docs_name",
       y = "sentiment")


bing_word_counts <- series %>%
  inner_join(get_sentiments("bing")) %>%
  dplyr::count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ggplot(aes(reorder(word, n), n, fill = sentiment)) +
  geom_bar(alpha = 0.8, stat = "identity", show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip()
