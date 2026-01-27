# ---------------------------------------------------------------------------#

import subprocess
import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------#

class CalamariPipeline:
    def __init__(self, model_path, image_dir, output_dir):
        self.model_path = model_path
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Creating the subdirectories.
        self.json_dir = self.output_dir / "json_outputs"
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures"
        
        for d in [self.json_dir, self.reports_dir, self.figures_dir]:
            d.mkdir(exist_ok=True)
    
    def find_images(self, extensions=('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')):
        """Finding all the images in the directories."""
        images = []
        for ext in extensions:
            images.extend(glob.glob(str(self.image_dir / ext)))
        return sorted(images)
    
    def run_batch_prediction(self, batch_size=10):
        """Running the predictions in batches!"""
        all_images = self.find_images()
        total_images = len(all_images)
        
        print(f"Found {total_images} images to process")

        for batch_num, i in enumerate(range(0, total_images, batch_size)):
            batch = all_images[i:i + batch_size]
            print(f"\nProcessing batch {batch_num + 1} ({len(batch)} images)...")
            
            cmd = [
                "calamari-predict",
                "--checkpoint", self.model_path,
                "--data.images"] + batch + [
                "--extended_prediction_data", "True",
                "--extended_prediction_data_format", "json",
                "--output_dir", str(self.json_dir),
                "--verbose", "False"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Warning: Batch {batch_num + 1} had errors:")
                    print(result.stderr[:500])
                else:
                    print(f"✓ Batch {batch_num + 1} completed")
            except subprocess.TimeoutExpired:
                print(f"✗ Batch {batch_num + 1} timed out")
            except Exception as e:
                print(f"✗ Error in batch {batch_num + 1}: {e}")
    
    def parse_json_outputs(self):
        """Parsing all JSON outputs and extracting statistics."""
        json_files = list(self.json_dir.glob("*.json"))
        print(f"\nFound {len(json_files)} JSON output files")
        
        all_data = []
        char_stats = defaultdict(list)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Getting a voted prediction + extracting data.
                voted_pred = data['predictions'][0]  # "id": "voted"

                image_name = Path(data['line_path']).name
                prediction = voted_pred['sentence']
                avg_confidence = voted_pred['avg_char_probability']
                
                # Character-level statistics!!
                char_details = []
                for pos in voted_pred['positions']:
                    if pos['chars']:
                        # Get top character
                        top_char = max(pos['chars'], key=lambda x: x['probability'])
                        char_details.append({
                            'character': top_char['char'],
                            'confidence': top_char['probability'],
                            'alternatives': [
                                {'char': c['char'], 'prob': c['probability']} 
                                for c in pos['chars'][:5]  # Top 5 alternatives.
                            ]
                        })
                
                # Collecting per-character statistics.
                for char_info in char_details:
                    char_stats[char_info['character']].append(char_info['confidence'])
                
                # Compiling image data.
                image_data = {
                    'image': image_name,
                    'prediction': prediction,
                    'avg_confidence': avg_confidence,
                    'length': len(prediction),
                    'chars': char_details
                }
                
                all_data.append(image_data)
                
            except Exception as e:
                print(f"Error parsing {json_file.name}: {e}")
        
        return all_data, char_stats
    
    def run_full_pipeline(self, batch_size=10):
        """Runs the complete pipeline."""
        print("=" * 80)
        print("STARTING CALAMARI OCR PIPELINE")
        print("=" * 80)
        
        # Step 1: Run batch predictions.
        print("\nSTEP 1: Running batch predictions...")
        self.run_batch_prediction(batch_size)
        
        # Step 2: Parse results.
        print("\nSTEP 2: Parsing JSON outputs...")
        all_data, char_stats = self.parse_json_outputs()
        
        if not all_data:
            print("No data to analyze!")
            return
        
        print("\n" + "=" * 80)
        print("JSON GENERATION COMPLETED!")
        print(f"JSON files saved to: {self.json_dir}")
        print("=" * 80)
        
        return all_data, char_stats

# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\bgat\Desktop\Antonis\ICPR26_model\*.json"
    IMAGE_DIR = r"C:\Users\bgat\Desktop\Antonis"
    OUTPUT_DIR = r"C:\Users\bgat\Desktop\Antonis\CALAMARI_PIPELINE_OUTPUT"
    
    # This creates the JSON files from 
    # scratch (calls the Calamari prediction).
    pipeline = CalamariPipeline(MODEL_PATH, IMAGE_DIR, OUTPUT_DIR)
    pipeline.run_full_pipeline(batch_size=5)
