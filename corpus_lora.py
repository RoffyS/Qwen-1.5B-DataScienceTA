import matplotlib.pyplot as plt
import json
import subprocess
from pathlib import Path
import argparse
from datetime import datetime

class LoraTrainingMonitor:
    def __init__(self, model_name, corpus_name):
        self.log_dir = Path(f"training_logs/{corpus_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.model_name = model_name
        self.corpus_name = corpus_name
        
    def start_training(self, num_iters=1):
        cmd = [
            "mlx_lm.lora",
            "--model", self.model_name,
            "--train",
            "--data", f"{self.corpus_name}",
            "--adapter-path", "lora_weights",
            "--iters", str(num_iters),
            "--learning-rate", "1e-4",
            "--max-seq-length", "8192",
            "--steps-per-eval", "50",  # Validation frequency
            "--val-batches", "50"      # Number of validation batches
            # "--resume-adapter-file", "lora_weights/adapters.safetensors"
        ]
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                self.parse_output(output.strip())
                print(output.strip())
                
        self.plot_curves()
        
    def parse_output(self, line):
        if "Train loss" in line:
            try:
                loss = float(line.split("Train loss")[1].split(',')[0].strip())
                self.train_losses.append(loss)
            except:
                pass
        elif "Val loss" in line:
            try:
                loss = float(line.split("Val loss")[1].split(',')[0].strip())
                self.val_losses.append(loss)
            except:
                pass
                
    def plot_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - {self.corpus_name}')
        plt.legend()
        plt.savefig(self.log_dir / 'learning_curves.png')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-coder-1.5b")
    parser.add_argument("--corpus", default="CorpusQAforTraining/combined")
    parser.add_argument("--iters", type=int, default=350)
    args = parser.parse_args()
    
    monitor = LoraTrainingMonitor(args.model, args.corpus)
    monitor.start_training(args.iters)