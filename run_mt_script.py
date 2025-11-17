import argparse
import concurrent.futures
import pandas as pd
import random
from openai import OpenAI
from agents import get_answer_cot, get_answer_MM, get_answer_Many, get_answer_zs_no_data
from Dataloader import UnifiedQADataLoader
from tqdm import tqdm

def process_iteration(i, dataloader, args, API_KEYS, base_url):
    try:
        # Dynamically select an API key from the list
        client = OpenAI(api_key=random.choice(API_KEYS),  base_url=base_url)
        agent_inputs = dataloader.get_agent_inputs(i)
        temp_results = {
            "question": agent_inputs["question"],
            "type": agent_inputs["type"],
            "modality": agent_inputs["modalities"],
            "true_answers": agent_inputs["answer"]
        }
        print("Processing index:", i)
        response = get_answer_MM(
            client,
            agent_inputs["question"],
            agent_inputs["text"],
            agent_inputs["table"],
            agent_inputs["images"],
            args.model  # model name variable
        )
        temp_results.update(response)
        # temp_results["predicted_answers"] = response
        print("Processed BITCH index:", i)
        return temp_results, None
    except Exception as e:
        return None, {"index": i, "error": str(e)}


def save_to_csv(results, errors, results_csv, errors_csv):
    pd.DataFrame(results).to_csv(results_csv, sep="^", index=False)
    pd.DataFrame(errors).to_csv(errors_csv, index=False)


def main(args):
    # Create the dataloader with the provided image dataset path.
    dataloader = UnifiedQADataLoader(
        dataset_type=args.dataset_type,
        dev_file=args.dev_file,
        tables_file=args.tables_file,
        texts_file=args.texts_file,
        images_file=args.images_file,
        images_base_url=args.images_base_url,
        encode_images=True
    )

    print("Data Loaded")
    
    # Use your API key here (consider using environment variables or configuration files for sensitive info)

    gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/"
    Openai_base_url = "https://api.openai.com/v1"

    client = OpenAI(api_key="MY_API_KEY")

    results = []  # To store successful responses
    errors = []   # To store errors with their iteration index

    # Create a thread pool for concurrent processing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_index = {
            executor.submit(process_iteration, i, dataloader, args, GEMINI_API_KEYS_T, gemini_base_url): i
            for i in tqdm(range(args.num_iterations))
        }

        # Process results as they complete.
        for future in tqdm(concurrent.futures.as_completed(future_to_index)):
            res, err = future.result()
            if res is not None:
                results.append(res)
            if err is not None:
                errors.append(err)
            # Save the current state after each job is done.
            save_to_csv(results, errors, args.results_csv, args.errors_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MultiModalQA data and get responses via the get_answer agent."
    )
    # Files and paths
    parser.add_argument("--dataset_type", type=str, default="multimqa",
                        help="Type of the Data.")
    parser.add_argument("--dev_file", type=str, default="./Datasets/MultiModalQA/Full_Multimodal_dev.jsonl",
                        help="Path to the development JSON file.")
    parser.add_argument("--tables_file", type=str, default="Datasets/MultiModalQA/MMQA_tables.jsonl",
                        help="Path to the tables JSONL file.")
    parser.add_argument("--texts_file", type=str, default="Datasets/MultiModalQA/MMQA_texts.jsonl",
                        help="Path to the texts JSONL file.")
    parser.add_argument("--images_file", type=str, default="Datasets/MultiModalQA/MMQA_images.jsonl",
                        help="Path to the images JSONL file.")
    parser.add_argument("--images_base_url", type=str, default="Datasets/MultiModalQA/final_dataset_images",
                        help="Base URL/path for images dataset.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Name of the model to use for get_answer.")
    parser.add_argument("--results_csv", type=str, default="result.csv",
                        help="Filename for saving the results CSV.")
    parser.add_argument("--errors_csv", type=str, default="errors.csv",
                        help="Filename for saving the errors CSV.")
    parser.add_argument("--num_iterations", type=int, default=12,
                        help="Number of iterations to process.")
    parser.add_argument("--num_threads", type=int, default=50,
                        help="Number of threads to use for concurrent processing.")

    args = parser.parse_args()
    main(args)
