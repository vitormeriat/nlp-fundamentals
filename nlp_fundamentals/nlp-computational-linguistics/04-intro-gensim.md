# Introducing Gensim 

So far, we haven't spoken much about finding hidden information - more about how to get our textual data in shape. We will be taking a brief departure from spaCy to discuss vector spaces and the open source Python package Gensim - this is because some of these concepts will be useful in the upcoming chapters and we would like to lay the foundation before moving on. However, we'll only be touching the surface of Gensim's capabilities. This chapter will introduce you to the data structures largely used in text analysis involving machine learning techniques - vectors [1]. This means that we are still in the domain of preprocessing and getting our data ready for further machine learning analysis. It may seem like overkill, focusing so much on just setting up our text/data, but like we've said before - garbage in, garbage out. While the previous chapter mostly involved text cleaning, we will be discussing converting our textual representations to numerical representations in this chapter, in particular, moving from strings to vectors.

When we talk about representations and transformations in this chapter, we will be exploring different kinds of ways of representing our strings as vectors, such as bag-of-words, TF-IDF (term frequency-inverse document frequency), LSI (latent semantic indexing), and the more recently popular word2vec. We will explain these methods soon on in Vectors and why we need them section and the rest in Chapter 8, Topic Models (Topic Modelling with Gensim) and Chapter 12, Word2Vec, Doc2Vec and Gensim, and gensim includes methods to do all of the above. The transformed vectors can be plugged into scikit-learn machine learning methods just as easily. Gensim started off as a modest project by Radim Rehurek and was largely the discussion of his Ph.D. thesis [17], Scalability of Semantic Analysis in Natural Language Processing [2]. It included novel implementations of Latent Dirichlet allocation [3] (LDA) and Latent Semantic Analysis [4] among its primary algorithms, as well as TF-IDF and Random projection [5] implementations. It has since grown to be one of the largest NLP/Information Retreival Python libraries, and is both memory-efficient and scalable, as opposed to the previous largely academic code available for semantic modelling (for example, the Stanford Topic Modelling Toolkit [6]).

Gensim manages to be scalable because it uses Python's in-built generators and iterators for streamed data-processing, so the data-set is never actually completely loaded in the RAM. Most IR algorithms involve matrix decompositions - which involve matrix multiplications. This is performed by NumPy, which is further built on FORTRAN/C, which is highly optimized for mathematical operations. Since all the heavy lifting is passed on to these low-level BLAS libraries, Gensim offers the ease-of-use of Python with the power of C.

The primary features of Gensim are its memory-independent nature, multicore implementations of latent semantic analysis, latent Dirichlet allocation, random projections, hierarchical Dirichlet process (HDP), and word2vec deep learning, as well as the ability to use LSA and LDA on a cluster of computers. It also seamlessly plugs into the Python scientific computing ecosystem and can be extended with other vector space algorithms. Gensim's directory of Jupyter notebooks [7] serves as an important documentation source, with its tutorials covering most of that Gensim has to offer. Jupyter notebooks are a useful way to run code on a live server - the documentation page [8] is worth having a look at!

The tutorials page can help you with getting started with using Gensim, but the coming sections will also describe how to get started with using Gensim, and about how important a role vectors will play in the rest of our time exploring machine learning and text processing.

## Vectors and why we need them

We're now moving toward the machine learning part of text analysis - this means that we will now start playing a little less with words and a little more with numbers. Even when we used spaCy, the POS-tagging and NER-tagging, for example, was done through statistical models - but the inner workings were largely hidden for us - we passed over Unicode text and after some magic, we have annotated text.

For Gensim however, we're expected to pass vectors as inputs to the IR algorithms (such as LDA or LSI), largely because what's going on under the hood is mathematical operations involving matrices. This means that we have to represent what was previously a string as a vector - and these kind of representations or models are called Vector Space Models [9].

From a mathematical perspective, a vector is a geometric object that has magnitude and direction. We don't need to pay as much attention to this, and rather think of vectors as a way of projecting words onto a mathematical space while preserving the information provided by these words.

Machine learning algorithms use these vectors to make predictions. We can understand machine learning as a suite of statistical algorithms and the study of these algorithms. The purpose of these algorithms is to learn from the provided data by decreasing the error of their predictions. As such, this is a wide field - we will be explaining particular machine learning algorithms as and then they come up.

Let's meanwhile discuss a couple of forms of these representations.

## Bag-of-words

The bag-of-words model is arguably the most straightforward form of representing a sentence as a vector. Let's start with an example:

```
S1:"The dog sat by the mat."
S2:"The cat loves the dog."
```

If we follow the same preprocessing steps we did in the Basic Preprocessing with language models section, from Chapter 3, spaCy's Language Models, we will end up with the following sentences:

```
S1:"dog sat mat."
S2:"cat love dog."
```

As Python lists, these will now look like this:

```
S1:['dog', 'sat', 'mat']
S2:['cat', 'love', 'dog']
```

If we want to represent this as a vector, we would need to first construct our vocabulary, which would be the unique words found in the sentences. Our vocabulary vector is now as follows:

```
Vocab = ['dog', 'sat', 'mat', 'love', 'cat']
```

This means that our representation of our sentences will also be vectors with a length of 5 - we can also say that our vectors will have 5 dimensions. We can also think of mapping of each word in our vocabulary to a number (or index), in which case we can also refer to our vocabulary as a dictionary. The bag-of-words model involves using word frequencies to construct our vectors. What will our sentences now look like?

```
S1:[1, 1, 1, 0, 0]
S2:[1, 0, 0, 1, 1]
```

It's easy enough to understand - there is 1 occurrence of dog, the first word in the vocabulary, and 0 occurrences of love in the first sentence, so the appropriate indexes are given the value based on the word frequency. If the first sentence has 2 occurrences of the word dog, it would be represented as:

```
S1: [2, 1, 1, 0, 0]
```

This is just an example of the idea behind a bag of words representation - the way Gensim approaches bag of words is slightly different, and we will see this in the coming section. One important feature of the bag-of-words model which we must remember is that it is an order less document representation - only the counts of the words matter. We can see that in our example above as well, where by looking at the resulting sentence vectors we do not know which words came first. This leads to a loss in spatial information, and by extension, semantic information. However, in a lot of information retrieval algorithms, the order of the words is not important, and just the occurrences of the words are enough for us to start with.

An example where the bag of words model can be used is in spam filtering - emails that are marked as spam are likely to contain spam-related words, such as buy, money, and stock. By converting the text in emails into a bag of words models, we can use Bayesian probability [10] to determine if it is more likely for a mail to be in the spam folder or not. This works because like we discussed before, in this case, the order of the words is not important - just whether they exist in the mail or not.

## TF-IDF

TF-IDF is short for term frequency-inverse document frequency. Largely used in search engines to find relevant documents based on a query, it is a rather intuitive approach to converting our sentences into vectors.

As the name suggests, TF-IDF tries to encode two different kinds of information - term frequency and inverse document frequency. Term frequency (TF) is the number of times a word appears in a document.

IDF helps us understand the importance of a word in a document. By calculating the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term) and then taking the logarithm of that quotient, we can have a measure of how common or rare the word is among all documents.

In case the preceding explanation wasn't very clear, expressing them as formulas will help!

- TF(t) = (number of times term t appears in a document) / (total number of terms in the document)
- IDF(t) = log_e (total number of documents / number of documents with term t in it)


TF-IDF is simply the product of these two factors - TF and IDF. Together it encapsulates more information into the vector representation, instead of just using the count of the words like in the bag-of-words vector representation. TF-IDF makes rare words more prominent and ignores common words such as is, of, and that, which may appear a lot of times, but have little importance.

For more information on how TF-IDF works, especially with the mathematical nature of TF-IDF and solved examples, the Wikipedia page [11] on TF-IDF is a good resource.








# References

- [1] Vectors: https://en.wikipedia.org/wiki/Euclidean_vector
- [2] Scalability of Semantic Analysis in Natural Language Processing: https://radimrehurek.com/phd_rehurek.pdf
- [3] Latent Dirichlet allocation: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
- [4] Latent semantic indexing: https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing
- [5] Random Projection: https://en.wikipedia.org/wiki/Random_projection
- [6] Stanford TMT: https://nlp.stanford.edu/software/tmt/tmt-0.4/
- [7] Gensim notebooks: https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks
- [8] Jupyter Notebooks: http://jupyter-notebook.readthedocs.io/en/stable/notebook.html
- [9] Vector Space Models: https://en.wikipedia.org/wiki/Vector_space_model
- [10] Bayesian Probability: https://en.wikipedia.org/wiki/Bayesian_probability
- [11] TF-IDF: https://en.wikipedia.org/wiki/Tf-idf
- [12] The Amazing power of word vectors: https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/
- [13] Corpora and Vector Spaces: https://radimrehurek.com/gensim/tut1.html
- [14] Bi-Gram example notebook: https://github.com/bhargavvader/personal/tree/master/notebooks/text_analysis_tutorial
- [15] N-grams: https://en.wikipedia.org/wiki/N-gram
- [16] Gensim dictionary: https://radimrehurek.com/gensim/corpora/dictionary.html
- [17] Scalability of Semantic Analysis in Natural Language Processing: https://radimrehurek.com/phd_rehurek.pdf
