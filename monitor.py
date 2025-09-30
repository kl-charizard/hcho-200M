"""
Training monitoring script
Monitors GPU usage, memory, and training progress
"""

import time
import psutil
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and system resources"""
    
    def __init__(self, log_file: str = "training.log"):
        self.log_file = log_file
        self.metrics = {
            'timestamps': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'cpu_usage': [],
            'ram_usage': [],
            'loss': [],
            'learning_rate': []
        }
    
    def get_gpu_info(self):
        """Get GPU information"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            return gpu_memory, gpu_utilization
        return 0, 0
    
    def get_system_info(self):
        """Get system resource usage"""
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        return cpu_usage, ram_usage
    
    def parse_training_log(self):
        """Parse training log for metrics"""
        if not Path(self.log_file).exists():
            return None, None
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Get latest loss and learning rate
            latest_loss = None
            latest_lr = None
            
            for line in reversed(lines):
                if "Loss=" in line and "LR=" in line:
                    parts = line.split("Loss=")[1].split(",")[0]
                    latest_loss = float(parts)
                    lr_part = line.split("LR=")[1].split(")")[0]
                    latest_lr = float(lr_part)
                    break
            
            return latest_loss, latest_lr
        except Exception as e:
            logger.error(f"Error parsing log: {e}")
            return None, None
    
    def update_metrics(self):
        """Update all metrics"""
        timestamp = time.time()
        
        # System metrics
        gpu_memory, gpu_utilization = self.get_gpu_info()
        cpu_usage, ram_usage = self.get_system_info()
        
        # Training metrics
        loss, lr = self.parse_training_log()
        
        # Store metrics
        self.metrics['timestamps'].append(timestamp)
        self.metrics['gpu_memory'].append(gpu_memory)
        self.metrics['gpu_utilization'].append(gpu_utilization)
        self.metrics['cpu_usage'].append(cpu_usage)
        self.metrics['ram_usage'].append(ram_usage)
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(lr)
        
        # Keep only last 1000 points
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
    
    def print_status(self):
        """Print current status"""
        if not self.metrics['timestamps']:
            print("No metrics available yet")
            return
        
        latest_idx = -1
        
        print("\n" + "="*50)
        print("📊 TRAINING MONITOR")
        print("="*50)
        
        # GPU Info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🖥️  GPU: {gpu_name}")
            print(f"   Memory: {self.metrics['gpu_memory'][latest_idx]:.2f} GB")
            print(f"   Utilization: {self.metrics['gpu_utilization'][latest_idx]:.1f}%")
        else:
            print("🖥️  GPU: Not available")
        
        # System Info
        print(f"💻 CPU Usage: {self.metrics['cpu_usage'][latest_idx]:.1f}%")
        print(f"🧠 RAM Usage: {self.metrics['ram_usage'][latest_idx]:.1f}%")
        
        # Training Info
        if self.metrics['loss'][latest_idx] is not None:
            print(f"📉 Loss: {self.metrics['loss'][latest_idx]:.4f}")
        if self.metrics['learning_rate'][latest_idx] is not None:
            print(f"📈 Learning Rate: {self.metrics['learning_rate'][latest_idx]:.2e}")
        
        print("="*50)
    
    def plot_metrics(self, save_path: str = "training_metrics.png"):
        """Plot training metrics"""
        if len(self.metrics['timestamps']) < 2:
            print("Not enough data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Convert timestamps to relative time
        start_time = self.metrics['timestamps'][0]
        times = [(t - start_time) / 3600 for t in self.metrics['timestamps']]  # Hours
        
        # GPU Memory
        axes[0, 0].plot(times, self.metrics['gpu_memory'])
        axes[0, 0].set_title('GPU Memory Usage')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Memory (GB)')
        axes[0, 0].grid(True)
        
        # Loss
        loss_data = [l for l in self.metrics['loss'] if l is not None]
        loss_times = [times[i] for i, l in enumerate(self.metrics['loss']) if l is not None]
        if loss_data:
            axes[0, 1].plot(loss_times, loss_data)
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # CPU Usage
        axes[1, 0].plot(times, self.metrics['cpu_usage'])
        axes[1, 0].set_title('CPU Usage')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Usage (%)')
        axes[1, 0].grid(True)
        
        # RAM Usage
        axes[1, 1].plot(times, self.metrics['ram_usage'])
        axes[1, 1].set_title('RAM Usage')
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Usage (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Metrics plot saved: {save_path}")
    
    def save_metrics(self, file_path: str = "metrics.json"):
        """Save metrics to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Metrics saved: {file_path}")
    
    def run_monitor(self, interval: int = 30):
        """Run continuous monitoring"""
        print("🔄 Starting training monitor...")
        print(f"   Update interval: {interval} seconds")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                self.update_metrics()
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            self.plot_metrics()
            self.save_metrics()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor LLM training")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    parser.add_argument("--log-file", type=str, default="training.log", help="Training log file")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_file)
    
    if args.plot_only:
        monitor.update_metrics()
        monitor.plot_metrics()
        monitor.save_metrics()
    else:
        monitor.run_monitor(args.interval)

if __name__ == "__main__":
    main()
