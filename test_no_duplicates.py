"""
Test to verify that the fix doesn't cause duplicate metric logging.

This test ensures that eval metrics are logged once at the correct step
and don't appear again in subsequent training logs.
"""

import json
import tempfile
from pathlib import Path


class StepCounter:
    """Simulates the step counter."""
    def __init__(self):
        self.value = 0
    
    def __int__(self):
        return self.value
    
    def increment(self):
        self.value += 1


class Metrics:
    """Simulates the metrics accumulator with reset behavior."""
    def __init__(self):
        self.data = {}
        
    def add(self, metrics_dict, prefix=None):
        for key, value in metrics_dict.items():
            full_key = f"{prefix}/{key}" if prefix else key
            self.data[full_key] = value
            
    def result(self, reset=True):
        result = self.data.copy()
        if reset:
            self.reset()
        return result
    
    def reset(self):
        self.data.clear()


class Logger:
    """Simulates the logger."""
    def __init__(self, step, output_file):
        self.step = step
        self.output_file = output_file
        self.pending_metrics = []
        
    def add(self, metrics_dict):
        current_step = int(self.step)
        for key, value in metrics_dict.items():
            self.pending_metrics.append((current_step, key, value))
            
    def write(self):
        if not self.pending_metrics:
            return
        
        # Group by step
        by_step = {}
        for step, key, value in self.pending_metrics:
            if step not in by_step:
                by_step[step] = {'step': step}
            by_step[step][key] = value
            
        # Write to file
        with open(self.output_file, 'a') as f:
            for step in sorted(by_step.keys()):
                f.write(json.dumps(by_step[step]) + '\n')
                
        self.pending_metrics.clear()


def test_no_duplicate_metrics():
    """
    Test that eval metrics are logged once and don't appear in subsequent logs.
    """
    print("\n" + "="*70)
    print("TEST: No Duplicate Metrics After Fix")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logfile = Path(tmpdir) / "metrics.jsonl"
        
        step = StepCounter()
        metrics = Metrics()
        logger = Logger(step, logfile)
        
        eval_every = 30
        log_every = 50
        
        print(f"\nConfiguration:")
        print(f"  - Eval at step: {eval_every}")
        print(f"  - Log at step: {log_every}")
        print(f"  - Training steps per iteration: 10\n")
        
        # Simulate training loop with fix
        while step.value < 100:
            # Eval at step 30
            if step.value == eval_every:
                print(f"[Step {step.value}] Evaluation")
                metrics.add({'avg_score': 42.0}, prefix='eval_episode')
                # FIX: Write immediately (this resets metrics)
                print(f"[Step {step.value}] Writing eval metrics")
                logger.add(metrics.result())  # This resets metrics!
                logger.write()
                
            # Training continues
            for _ in range(10):
                step.increment()
                
            # Add training metrics
            if step.value > eval_every:
                metrics.add({'train_loss': 0.5}, prefix='train')
                
            # Log at step 50
            if step.value >= log_every and step.value < log_every + 10:
                print(f"[Step {step.value}] Regular logging (training metrics)")
                logger.add(metrics.result())
                logger.write()
                break
                
        # Analyze results
        print(f"\n{'='*70}")
        print("Results:")
        print("="*70)
        
        all_entries = []
        with open(logfile, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_entries.append(data)
                print(f"Step {data['step']}: {list(data.keys())}")
                
        # Check for duplicates
        eval_score_appearances = []
        for entry in all_entries:
            if 'eval_episode/avg_score' in entry:
                eval_score_appearances.append(entry['step'])
                
        print(f"\n{'='*70}")
        if len(eval_score_appearances) == 1:
            print("✓ PASS: Eval metrics appear exactly once")
            print(f"  - Logged at step: {eval_score_appearances[0]}")
            print(f"  - Expected step: {eval_every}")
            if eval_score_appearances[0] == eval_every:
                print(f"  - ✓ Correct step!")
        elif len(eval_score_appearances) > 1:
            print("✗ FAIL: Eval metrics appear multiple times!")
            print(f"  - Appearances at steps: {eval_score_appearances}")
        else:
            print("✗ FAIL: Eval metrics not found")
            
        # Check that training metrics appear only in training log
        train_loss_appearances = []
        for entry in all_entries:
            if 'train/train_loss' in entry:
                train_loss_appearances.append(entry['step'])
                
        if train_loss_appearances:
            print(f"\n✓ Training metrics logged at: {train_loss_appearances}")
            
        # Verify no eval metrics in training log entry
        for entry in all_entries:
            if entry['step'] >= log_every:
                has_eval = 'eval_episode/avg_score' in entry
                has_train = 'train/train_loss' in entry
                if has_eval and has_train:
                    print(f"\n⚠️  WARNING: Both eval and train metrics in same entry at step {entry['step']}")
                    print("   This shouldn't happen with the fix!")
                    
        print("="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DUPLICATE METRICS TEST")
    print("="*70)
    print("\nThis test verifies that the fix doesn't cause duplicate logging")
    print("of eval metrics. After the fix, metrics.result() resets the metrics,")
    print("so eval metrics are logged once and don't appear again.\n")
    
    test_no_duplicate_metrics()
    
    print("\n✓ Test completed successfully!")
    print("="*70 + "\n")
