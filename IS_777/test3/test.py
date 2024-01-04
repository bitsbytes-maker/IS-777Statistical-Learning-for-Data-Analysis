import corpus
import vocabulary 
import numpy as np
import gDMR 
import glda


def process_in_chunks(model_class, docfilepath, vecfilepath, chunk_size, G, sigma, beta, voca, iteration):
    # Function to split the dataset into chunks
    def split_corpus(corpus, chunk_size):
        # Convert the Corpus object into a list of documents 
        doc_list = list(corpus)
        for i in range(0, len(doc_list), chunk_size):
            yield doc_list[i:i + chunk_size]


    # Read the full datasets
    full_corpus = corpus.Corpus.read(docfilepath)
    full_vecs_dataset = corpus.Corpus.read(vecfilepath, dtype=float)
    full_vecs = np.array([[v for v in vec] for vec in full_vecs_dataset], dtype=np.float64)

    # Initialize variables to hold combined results
    combined_word_dist = None

    # Process each chunk
    for chunk_idx, chunk in enumerate(split_corpus(full_corpus, chunk_size)):
        # Determine the range of feature vectors corresponding to the current chunk
        vec_range = slice(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size)
        vecs_chunk = full_vecs[vec_range]

        # Convert chunk to numerical format
        docs = [voca.doc_to_ids(doc) for doc in chunk]

        # Initialize the model for this chunk
        model = model_class(G, sigma, beta, docs, vecs_chunk, voca.size())

        # Train the model on this chunk
        model.learning(iteration, voca)

        # Aggregate the results
        chunk_word_dist = model.word_dist_with_voca(voca)
        combined_word_dist = update_combined_results(combined_word_dist, chunk_word_dist)

    # Return the aggregated results
    return combined_word_dist

def update_combined_results(combined_results, new_results):
    # Implement logic to combine new_results into combined_results
    # This could involve averaging probabilities, summing counts, etc.
    if combined_results is None:
        return new_results
    else:
        # Combine the results for each topic
        for topic in new_results:
            if topic in combined_results:
                for word, prob in new_results[topic].items():
                    if word in combined_results[topic]:
                        combined_results[topic][word] += prob  # Example: summing probabilities
                    else:
                        combined_results[topic][word] = prob
            else:
                combined_results[topic] = new_results[topic]
        return combined_results


chunk_size = 1000  # Define the size of each chunk
G = 20  # number of topics
sigma = 1.0  # hyperparameter
beta = 0.01  # hyperparameter                                             
voca = vocabulary.Vocabulary()  # Initialize vocabulary
iteration = 100  # Number of iterations for learning

combined_results = process_in_chunks(gDMR.gDMR, 'text.txt', 'covariates.txt', chunk_size, G, sigma, beta, voca, iteration)

# New code to save results to a file
output_filename = 'combined_results.txt'
with open(output_filename, 'w') as file:
    for k in combined_results:
        file.write(f"GROUP {k}\n")
        top_words_and_probs = sorted(combined_results[k].items(), key=lambda x: x[1], reverse=True)[:10]
        for word, prob in top_words_and_probs:
            file.write(f"{word}: {prob:.2f}\n")
        file.write("\n")

print(f"Results saved to {output_filename}")

# You can later download this file using your Python environment's file download functionality.