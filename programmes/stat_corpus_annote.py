import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')

def entrainement(test_size):
    french_stop_words = stopwords.words('french')
    corpus_dataframe = pd.read_csv('corpus_annote.csv', header=0, usecols=[2,4], names=['phrase', 'accord'])
    corpus_dataframe = corpus_dataframe.dropna(subset=['phrase'])

    train_corpus, test_corpus = train_test_split(corpus_dataframe, test_size=test_size, random_state=42, stratify=corpus_dataframe['accord'])
    model = make_pipeline(TfidfVectorizer(stop_words=french_stop_words), MultinomialNB())
    model.fit(train_corpus['phrase'], train_corpus['accord'])

    test_evaluation = model.predict(test_corpus['phrase'])
    accuracy = accuracy_score(test_corpus['accord'], test_evaluation)
    f1_macro = f1_score(test_corpus['accord'], test_evaluation, average='macro')
    f1_weighted = f1_score(test_corpus['accord'], test_evaluation, average='weighted')

    return accuracy, f1_macro, f1_weighted

def plot_results(test_sizes, accuracies, f1_macros, f1_weighteds):
    bar_width = 0.25
    r1 = np.arange(len(accuracies))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(12, 6))
    plt.bar(r1, accuracies, color='b', width=bar_width, edgecolor='grey', label='Accuracy')
    plt.bar(r2, f1_macros, color='c', width=bar_width, edgecolor='grey', label='F1 Score Macro')
    plt.bar(r3, f1_weighteds, color='m', width=bar_width, edgecolor='grey', label='F1 Score Weighted')
    plt.xlabel('Test Size', fontweight='bold', fontsize=15)
    plt.ylabel('Performance', fontweight='bold', fontsize=15)
    plt.xticks([r + bar_width for r in range(len(accuracies))], test_sizes)
    plt.title('Performance Comparison for Different Test Sizes')
    plt.legend()
    plt.show()

def main():
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accuracies = []
    f1_macros = []
    f1_weighteds = []

    for size in test_sizes:
        accuracy, f1_macro, f1_weighted = entrainement(test_size=size)
        accuracies.append(accuracy)
        f1_macros.append(f1_macro)
        f1_weighteds.append(f1_weighted)
    
    plot_results(test_sizes, accuracies, f1_macros, f1_weighteds)

main()
