"""
IoT-Shield AI Trainer - CLI version
===================================
This script runs the training process from the command line,
useful for when you can't or don't want to use the GUI.
It loads file paths from session.json and uses the same 
DataLoader and logic as the main app.
"""

import os
import sys
import json
import time
from data_loader import DataLoader
from trainer import TrainerThread

# Redirect signals to console for CLI usage
class CLITrainer(TrainerThread):
    def __init__(self, data_loader):
        super().__init__(data_loader=data_loader)
        # We override signals to print directly
        # Since we're in CLI, we don't need PyQt signals, 
        # but the run() method uses .emit(), so we'll mock them.
        
        class MockSignal:
            def emit(self, msg):
                try:
                    if isinstance(msg, dict):
                        print(f"\n[SUCCESS] Training completed with accuracy: {msg.get('accuracy', 0):.4f}")
                    elif isinstance(msg, int):
                        print(f"\r[PROGRESS] {msg}% ", end="", flush=True)
                    else:
                        # Print normally, if it fails, try encoding differently
                        try:
                            print(msg)
                        except UnicodeEncodeError:
                            print(msg.encode('ascii', errors='replace').decode('ascii'))
                except Exception as e:
                    pass

        self.progress_updated = MockSignal()
        self.log_message = MockSignal()
        self.training_completed = MockSignal()
        self.training_failed = MockSignal()

    def _cancel(self):
        print("\n[CANCELLED] Training stopped.")
        super()._cancel()

def main():
    session_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session.json")
    
    if not os.path.exists(session_file):
        print(f"Error: {session_file} not found. Please run the GUI once to select files or create it manually.")
        return

    with open(session_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        file_paths = data.get('file_paths', [])

    if not file_paths:
        print("Error: No file paths found in session.json.")
        return

    print(f"Starting CLI Retraining with {len(file_paths)} files...")
    
    dl = DataLoader()
    
    def cli_log(m):
        try:
            print(f"  {m}")
        except UnicodeEncodeError:
            print(f"  {str(m).encode('ascii', errors='replace').decode('ascii')}")

    print("Loading data previews and detecting columns...")
    try:
        dl.load_csv_files(file_paths, log_callback=cli_log)
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    if not dl.label_column:
        print("Error: No label column detected. Cannot train.")
        return

    print(f"\nReady to train. Label: {dl.label_column}, Features: {len(dl.feature_columns)}")
    
    trainer = CLITrainer(dl)
    
    print("\n" + "="*50)
    print("RUNNING INCREMENTAL TRAINING")
    print("="*50)
    
    start_time = time.time()
    trainer.run()
    end_time = time.time()
    
    print(f"\n\nAll done! Total time: {end_time - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
