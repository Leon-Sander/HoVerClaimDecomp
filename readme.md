# Master's Thesis Repository: Claim Verification Enhancement

## Introduction
This repository contains all the research materials and code associated with my master's thesis on enhancing the process of claim verification.

## Research Objective
The objective of this study is to advance the field of claim verification by applying claim decomposition strategies and enhancing retrieval mechanisms using a custom-trained embedding model. This involves simplifying complex multi-hop claims into more manageable components and refining them iteratively to enhance the accuracy of information retrieval systems.


# Repository File Descriptions

## Data Setup
- **Data Folder**: Before running the scripts, download the `hover dev` and `train` datasets along with the `wiki_wo_links.db` and save them into the `data` folder.

## Python Scripts
- **embed_all_documents.py**: Creates the `chroma_db_mistral` with HNSW and cosine similarity, embedding all documents from the `wiki_wo_links.db` using the E5-Mistral-7B-Instruct embedder.

- **create_randomized_claim_sample.py**: Includes functions to create a sample dataset or convert the full dataset into the required structure for further processing.

- **create_sentences_db.py**: Generates a copy of `wiki_wo_links.db` but with two additional columns: one containing the document split into sentences and another with the document title prepended to each sentence.

- **semantic_similarity_analysis.py**: Calculates the semantic similarity of decomposed claims and multi-hop claims to relevant facts for those claims.

- **cross_encoder/train_cross_encoder.py**: Initiates training of the cross encoder. Evaluation is handled by `evaluate_cross_encoder.py`.

- **cross_encoder/calculate_sentence_number_threshold.py**: Outputs the threshold evaluation results.

- **evaluate_retrieval.py**: Contains the `is_successful_retrieval(obj, retrieval_key)` function, which checks if the list in `obj[retrieval_key]` contains all relevant facts from `obj["supporting_facts"]`.

- **result_generation.ipynb**: Notebook with functions that evaluate different retrieval results and output them directly in LaTeX format.

## Additional Tools and Models
- **huggingface_llm_loading.py**: Contains the `TransformerLLM` class, which loads the Mistral LLM and includes functions for decomposition, subquestion generation, and claim refinement.

- **baseline_decomposition.py**: Generates the baseline decomposition for claims.

- **iterative_claim_refinement.py**: Executes the iterative refinement pipeline on the entire dataset structured by `create_randomized_claim_sample.py`.

- **cross_enc_threshold_refinement.py**: Refines claims with the top `n` sentences based on documents retrieved for the claim.

## System Requirements
- **GPU Requirements**: To run the pipeline, two GPUs are necessary.

## Running the Pipeline
Ensure you have the necessary hardware and data setup as described above. 