import random

# Sample data for chunks (retrieval context) and queries
chunks = {
    1: "The quick brown fox jumps over the lazy dog.",
    2: "In a galaxy far, far away, there is a battle between good and evil.",
    3: "Python is a versatile programming language used in various domains.",
    4: "The Mona Lisa is a famous painting by Leonardo da Vinci.",
    5: "Elephants are the largest land animals on Earth.",
    6: "The Amazon rainforest is home to a diverse range of flora and fauna.",
}

# Sample data for user queries
queries = [
    "Tell me about foxes.",
    "What's the story in the distant galaxy?",
    "Explain the features of Python programming language.",
    "Tell me about famous artworks.",
    "What is the size of an elephant?",
    "What animals live in the Amazon rainforest?",
]

# Create a ground truth  with two relevant chunks for each query
relevant_chunks_ground_truth = {
    "Tell me about foxes.": [1, 5],
    "What's the story in the distant galaxy?": [2, 6],
    "Explain the features of Python programming language.": [3, 1],
    "Tell me about famous artworks.": [4, 2],
    "What is the size of an elephant?": [5, 1],
    "What animals live in the Amazon rainforest?": [6, 5],
}

# Retrieval Evaluation
def retrieval_evaluation(chunks, queries, relevant_chunks_dict, k=2):
    hit_count = 0
    reciprocal_ranks = []
    context_precision_scores = []
    context_recall_scores = []

    retrieval_results = {}
    for query in queries:
        relevant_chunks = relevant_chunks_dict[query]
        retrieved_chunks = random.sample(list(chunks.keys()), min(k, len(chunks)))  # Simulate retrieval

        hit_count += any(chunk in relevant_chunks for chunk in retrieved_chunks)

        reciprocal_rank = 0
        for i, chunk in enumerate(retrieved_chunks):
            if chunk in relevant_chunks:
                reciprocal_rank = 1 / (i + 1)
                break

        reciprocal_ranks.append(reciprocal_rank)

        # Calculate Context Precision
        true_positives = len(set(retrieved_chunks).intersection(relevant_chunks))
        false_positives = len(set(retrieved_chunks).difference(relevant_chunks))
        context_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        context_precision_scores.append(context_precision)

        # Calculate Context Recall
        true_positives = len(set(retrieved_chunks).intersection(relevant_chunks))
        false_negatives = len(relevant_chunks) - true_positives
        context_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        context_recall_scores.append(context_recall)

        retrieval_results[query] = {
            "relevant_chunks": relevant_chunks,
            "retrieved_chunks": retrieved_chunks,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

    hit_rate = hit_count / len(queries)
    mrr = sum(reciprocal_ranks) / len(queries)
    avg_context_precision = sum(context_precision_scores) / len(queries)
    avg_context_recall = sum(context_recall_scores) / len(queries)

    return retrieval_results, hit_rate, mrr, avg_context_precision, avg_context_recall

retrieval_results, hit_rate, mrr, avg_context_precision, avg_context_recall = retrieval_evaluation(
    chunks, queries, relevant_chunks_ground_truth
)
print("Retrieval Results:")
print(retrieval_results)
print("\nHit Rate:", hit_rate)
print("MRR:", mrr)
print("Avg Context Precision:", avg_context_precision)
print("Avg Context Recall:", avg_context_recall)

# Response (Faithfulness) and Relevancy Evaluation

def faithfulness_relevancy_evaluation(generated_responses, relevant_chunks_dict):
    faithfulness_results = {}
    relevancy_results = {}

    for query, generated_response in generated_responses.items():
        relevant_chunks_for_query = relevant_chunks_dict[query]

        is_faithful = generated_response in relevant_chunks_for_query
        faithfulness_results[query] = {"is_faithful": is_faithful}

        is_relevant = any(chunk in relevant_chunks_for_query for chunk in generated_response.lower().split())
        relevancy_results[query] = {"is_relevant": is_relevant}

    return faithfulness_results, relevancy_results

# Simulate generated responses
generated_responses = {
    "Tell me about foxes.": "Foxes are small to medium-sized mammals known for their agility.",
    "What's the story in the distant galaxy?": "In a distant galaxy, a battle between good and evil unfolds.",
    "Explain the features of Python programming language.": "Python is known for its readability, versatility, and extensive libraries.",
    "Tell me about famous artworks.": "The Mona Lisa, created by Leonardo da Vinci, is a masterpiece.",
    "What is the size of an elephant?": "Elephants are the largest land animals, with males weighing up to 14,000 pounds.",
    "What animals live in the Amazon rainforest?": "The Amazon rainforest is home to diverse species, including jaguars, parrots, and toucans.",
}

faithfulness_results, relevancy_results = faithfulness_relevancy_evaluation(generated_responses, relevant_chunks_ground_truth)
print("\nFaithfulness Results:")
print(faithfulness_results)
print("\nRelevancy Results:")
print(relevancy_results)
