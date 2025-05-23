"""
Complete pipeline runner for the spam classifier.
"""

import subprocess
import sys
import os


def run_command(script_name):
    """Run a Python script and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run([sys.executable, script_name],
                                check=True,
                                capture_output=True,
                                text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Run the complete pipeline."""
    print("SPAM CLASSIFIER PIPELINE")
    print("This will run the complete pipeline: data loading, preprocessing, and training")

    scripts = [
        "step/load_data.py",
        "step/preprocess_data.py",
        "step/train_model.py"
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"ERROR: Script {script} not found!")
            sys.exit(1)

        success = run_command(script)
        if not success:
            print(f"Pipeline failed at {script}")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("You can now run the UI application with: python app_ui.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()