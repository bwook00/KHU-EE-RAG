import os.path
from pathlib import Path
from dotenv import load_dotenv

from autorag.deploy import GradioRunner

load_dotenv()

def main():
    root_dir = Path.cwd().resolve().parent

    # trial_path = os.path.join(root_dir, "results_4", "4")
    trial_path = os.path.join(root_dir, "results_upstage", "1")

    runner = GradioRunner.from_trial_folder(trial_path)
    runner.run_web(server_name="0.0.0.0", server_port=7680, share=True)


if __name__ == "__main__":
    main()

