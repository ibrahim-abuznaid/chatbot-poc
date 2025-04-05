import os
import json
from langsmith import Client, evaluate
import openai
from langsmith import wrappers
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import system functions from our two apps
from app_v3 import process_query, prepare_documents, create_vector_store
from no_rag import get_answer

# Ensure environment variables are set
os.environ["LANGSMITH_TRACING"] = "true"
os.environ.setdefault("OPENAI_API_KEY", "your_openai_api_key_here")

# Wrap OpenAI client for LLM calls (both for dataset generation and evaluation)

openai_client = wrappers.wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

def generate_qa_dataset(num_pairs=1):
    """
    Use an LLM to generate Q/A pairs related to building codes and regulations.
    Expect each pair to be on its own line formatted like:
      Q: <question> | A: <answer>
    """
    prompt = (
        f"Generate {num_pairs} question and answer pairs about building codes and regulations. "
        "Format each pair on a new line as 'Q: <question> | A: <answer>'."
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an expert in building codes and regulations."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content.strip()
    
    pairs = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            try:
                question_part, answer_part = line.split("|", 1)
                question = question_part.replace("Q:", "").strip()
                answer = answer_part.replace("A:", "").strip()
                if question and answer:
                    pairs.append((question, answer))
            except Exception as e:
                print(f"Error parsing line: {line} -> {e}")
    return pairs

# Define an LLM-as-judge evaluator that compares the system's answer to the reference.
default_eval_instructions = (
    "You are a teacher grading answers for questions about building codes and regulations. "
    "Given the question, the reference answer, and the system's answer, "
    "respond with CORRECT if the system's answer is factually accurate relative to the reference answer, "
    "or INCORRECT otherwise."
)

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    prompt = (
        f"Question: {inputs['question']}\n"
        f"Reference Answer: {reference_outputs['answer']}\n"
        f"System Answer: {outputs['response']}\n"
        "Grade as CORRECT or INCORRECT:"
    )
    result = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": default_eval_instructions},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message.content.strip()
    return result.upper() == "CORRECT"

# Define target functions for our evaluation experiments

def target_app_v3_basic(example: dict) -> dict:
    result = process_query(example["question"], qtt="basic", llm_model="openai")
    return {"response": result.get("response", "")}

def target_app_v3_fusion(example: dict) -> dict:
    # Evaluate using qtt "fusion"
    result = process_query(example["question"], qtt="fusion", llm_model="openai")
    return {"response": result.get("response", "")}

def target_app_v3_decomposition(example: dict) -> dict:
    # Evaluate using qtt "decomposition"
    result = process_query(example["question"], qtt="decomposition", llm_model="openai")
    return {"response": result.get("response", "")}

def target_app_v3_hyde(example: dict) -> dict:
    # Evaluate using qtt "hyde"
    result = process_query(example["question"], qtt="hyde", llm_model="openai")
    return {"response": result.get("response", "")}

def target_no_rag(example: dict) -> dict:
    # The no-rag system loads its own document and provides an answer.
    answer = get_answer(example["question"], None)
    return {"response": answer}

# Add this function to create and save the initial vector store
def initialize_vector_store(docs_path="docx.md", save_path="faiss_index"):
    """Initialize and save vector store from documents"""
    try:
        # Load documents
        loader = TextLoader(docs_path, encoding="utf-8")
        documents = loader.load()
        
        # Prepare documents
        document_splits = prepare_documents(documents)
        
        # Create vector store
        vectorstore = create_vector_store(document_splits)
        
        # Save vector store
        vectorstore.save_local(save_path)
        print(f"Vector store saved to {save_path}")
        
        # Verify loading works
        test_load = FAISS.load_local(
            save_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Successfully verified vector store loading")
        
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise

def main():
    # Generate Q/A pairs
    qa_pairs = generate_qa_dataset(num_pairs=5)
    if not qa_pairs:
        print("Failed to generate Q/A dataset.")
        return

    # Create evaluation data
    inputs = [{"question": q} for q, a in qa_pairs]
    reference_outputs = [{"answer": a} for q, a in qa_pairs]

    # Disable LangSmith tracing for local evaluation
    os.environ["LANGSMITH_TRACING"] = "false"

    # Run evaluations for each QTT variant
    qtt_variants = {
        "basic": target_app_v3_basic,
        "fusion": target_app_v3_fusion,
        "decomposition": target_app_v3_decomposition,
        "hyde": target_app_v3_hyde,
        "no_rag": target_no_rag
    }

    print("\nRunning Evaluations:")
    results = {}
    
    for qtt_name, target_func in qtt_variants.items():
        print(f"\nEvaluating {qtt_name}...")
        try:
            # Run evaluation for each example manually
            qtt_results = []
            for i, (input_data, ref_output) in enumerate(zip(inputs, reference_outputs)):
                try:
                    # Get system response
                    system_output = target_func(input_data)
                    
                    # Run evaluator
                    is_correct = correctness_evaluator(
                        input_data, 
                        system_output,
                        ref_output
                    )
                    
                    qtt_results.append({
                        "example_id": i,
                        "question": input_data["question"],
                        "system_answer": system_output["response"],
                        "reference_answer": ref_output["answer"],
                        "is_correct": is_correct
                    })
                    
                except Exception as e:
                    print(f"Error evaluating example {i}: {str(e)}")
                    continue
                    
            results[qtt_name] = qtt_results
            
        except Exception as e:
            print(f"Error evaluating {qtt_name}: {str(e)}")
            continue

    # Print results summary
    print("\n=== Evaluation Results ===")
    for qtt_name, qtt_results in results.items():
        correct_count = sum(1 for r in qtt_results if r["is_correct"])
        total_count = len(qtt_results)
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\n{qtt_name.upper()} Results:")
        print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)")
        
        print("\nDetailed Results:")
        for r in qtt_results:
            print(f"\nQuestion: {r['question']}")
            print(f"System: {r['system_answer']}")
            print(f"Reference: {r['reference_answer']}")
            print(f"Correct: {r['is_correct']}")

    # Optional: Create pandas DataFrame
    try:
        import pandas as pd
        all_results = []
        for qtt_name, qtt_results in results.items():
            for r in qtt_results:
                r["qtt"] = qtt_name
                all_results.append(r)
        
        df = pd.DataFrame(all_results)
        print("\nResults Summary (DataFrame):")
        print(df.groupby("qtt")["is_correct"].agg(["count", "mean", "sum"]))
        
    except ImportError:
        print("\nInstall pandas for additional analysis")

if __name__ == "__main__":
    # Initialize vector store if it doesn't exist
    if not os.path.exists("faiss_index"):
        print("Initializing vector store...")
        initialize_vector_store()
    main() 