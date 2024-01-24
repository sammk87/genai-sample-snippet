import os
import re
import pandas as pd
from bert_score import BERTScorer

# Sample data
text = "Python is a versatile programming language used in various domains. It is known for its readability and ease of use. Many developers appreciate Python for its simplicity and the availability of a vast ecosystem of libraries."

# Three different summaries of the text
summary_1 = "Python is a versatile and easy-to-use programming language with widespread use in various domains. Its simplicity and readability make it a preferred choice for many developers."
summary_2 = "Known for its versatility, Python is a programming language widely used across different domains. Developers favor Python for its readability and the extensive set of libraries available."
summary_3 = "Python, a programming language renowned for its simplicity and readability, finds applications in diverse domains. Developers appreciate its versatility and the rich ecosystem of libraries."

def get_rouge_scores(candidate, reference, n=1):
    """
    Calculate precision, recall, and F1 score for n-grams overlap between candidate and reference.
    
    Args:
    - candidate (str): Candidate text.
    - reference (str): Reference text.
    - n (int): n-grams size.
    
    Returns:
    - Tuple of precision, recall, and F1 score.
    """
    def get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return ngrams

    def calculate_overlap(candidate_ngrams, reference_ngrams):
        overlap_count = len(set(candidate_ngrams) & set(reference_ngrams))
        return overlap_count

    candidate_tokens = candidate.split()
    reference_tokens = reference.split()

    # Get n-grams for candidate and reference
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    reference_ngrams = get_ngrams(reference_tokens, n)

    # Calculate overlap
    overlap = calculate_overlap(candidate_ngrams, reference_ngrams)

    # Calculate precision, recall, and F1 score
    precision = overlap / len(candidate_ngrams)
    recall = overlap / len(reference_ngrams)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1_score

def rouge_evaluation(candidate, reference):
    """
    Evaluate ROUGE scores for unigram, bigram, and longest common subsequence (LCS) between candidate and reference.
    
    Args:
    - candidate (str): Candidate text.
    - reference (str): Reference text.
    
    Returns:
    - Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    rouge_1_scores = get_rouge_scores(candidate, reference, n=1)
    rouge_2_scores = get_rouge_scores(candidate, reference, n=2)
    rouge_l_scores = get_rouge_scores(candidate, reference, n=min(len(candidate.split()), len(reference.split())))

    return {
        "ROUGE-1 Precision": rouge_1_scores[0],
        "ROUGE-1 Recall": rouge_1_scores[1],
        "ROUGE-1 F1 Score": rouge_1_scores[2],
        "ROUGE-2 Precision": rouge_2_scores[0],
        "ROUGE-2 Recall": rouge_2_scores[1],
        "ROUGE-2 F1 Score": rouge_2_scores[2],
        "ROUGE-L Precision": rouge_l_scores[0],
        "ROUGE-L Recall": rouge_l_scores[1],
        "ROUGE-L F1 Score": rouge_l_scores[2]
    }

# ...

print(rouge_evaluation(text, summary_1))


# Instantiate the BERTScorer object for English language
scorer = BERTScorer(lang="en")

# Calculate BERTScore for the summary 1 against the excerpt
P1, R1, F1_1 = scorer.score([text], [summary_1])
print("Summary 1 BERTScore F1 Score:", F1_1.tolist()[0])
