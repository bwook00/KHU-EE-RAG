from dotenv import load_dotenv
from pathlib import Path
import os

import autorag
from autorag.evaluator import Evaluator
from autorag.embedding.base import embedding_models
from autorag import LazyInit
from llama_index.llms.upstage import Upstage
from llama_index.embeddings.upstage import UpstageEmbedding

load_dotenv()

def main():
    # Set api key
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")

    # Custom Embedding Model Register
    embedding_models['upstage'] = LazyInit(UpstageEmbedding, api_key=upstage_api_key, embed_batch_size=1)

    # Custom LLM Model Register
    autorag.generator_models['upstage'] = Upstage

    # Define paths
    root_dir = Path.cwd().resolve().parent

    qa_path = os.path.join(root_dir, 'data', 'qa', 'duplicate_pa.parquet')
    corpus_dir = os.path.join(root_dir, 'data', 'corpus', 'duplicate_file_name.parquet')
    project_dir = os.path.join(root_dir, 'results_upstage')
    yaml_dir = os.path.join(root_dir, 'config', 'config_upstage_embedding.yaml')

    # Run Evaluation
    evaluator = Evaluator(qa_data_path=qa_path,
                          corpus_data_path=corpus_dir,
                          project_dir=project_dir)

    evaluator.start_trial(yaml_dir, skip_validation=True)
    # evaluator.restart_trial(os.path.join(project_dir, '1'))


if __name__ == "__main__":
    main()
