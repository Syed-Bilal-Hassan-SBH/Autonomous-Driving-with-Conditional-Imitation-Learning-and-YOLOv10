#!/usr/bin/env python3
"""
Create comprehensive result figures and visualizations
Generate publication-ready plots for all experiments
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultFiguresGenerator:
    """Generate comprehensive result figures"""
    
    def __init__(self):
        self.base_dir = Path("c:/Users/MASTER/Desktop/Sem 07/CV/Self-Driving Cars/YOLOv10_Autonomous_Driving")
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def load_all_results(self):
        """Load all experiment results"""
        results = {}
        
        for dataset in ["CARLA", "Udacity"]:
            results[dataset] = {}
            
            # Original paper results
            try:
                with open(self.results_dir / dataset.lower() / "original_paper_trained_100epochs" / "training_results.json", 'r') as f:
                    results[dataset]['original_paper'] = json.load(f)
            except:
                logger.warning(f"Could not load original paper results for {dataset}")
            
            # YOLO testing results
            try:
                with open(self.results_dir / dataset.lower() / "yolov10_pretrained" / "testing_results.json", 'r') as f:
                    results[dataset]['yolo_testing'] = json.load(f)
            except:
                logger.warning(f"Could not load YOLO testing results for {dataset}")
            
            # YOLO fine-tuning results
            try:
                with open(self.results_dir / dataset.lower() / "yolov10_finetuned_100epochs" / "training_results.json", 'r') as f:
                    results[dataset]['yolo_finetuned'] = json.load(f)
            except:
                logger.warning(f"Could not load YOLO fine-tuning results for {dataset}")
        
        # Ablation studies
        try:
            with open(self.results_dir / "ablation_studies_comprehensive" / "ablation_results.json", 'r') as f:
                results['ablation'] = json.load(f)
        except:
            logger.warning("Could not load ablation study results")
        
        return results
    
    def create_training_progression_figure(self, results):
        """Create training progression comparison figure"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Progression Across Models and Datasets', fontsize=16, fontweight='bold')
        
        # CARLA - Original Paper
        ax = axes[0, 0]
        if 'CARLA' in results and 'original_paper' in results['CARLA']:
            history = results['CARLA']['original_paper']['training_history']
            epochs = [h['epoch'] for h in history]
            train_loss = [h['train_loss'] for h in history]
            val_loss = [h['val_loss'] for h in history]
            
            ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('CARLA - Original Paper Model')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # CARLA - YOLO Models
        ax = axes[0, 1]
        if 'CARLA' in results:
            if 'yolo_testing' in results['CARLA']:
                history = results['CARLA']['yolo_testing']['training_history']
                epochs = [h['epoch'] for h in history]
                map_scores = [h['map50'] for h in history]
                ax.plot(epochs, map_scores, 'g-', label='YOLOv10 Pretrained', linewidth=2, alpha=0.8)
            
            if 'yolo_finetuned' in results['CARLA']:
                history = results['CARLA']['yolo_finetuned']['training_history']
                epochs = [h['epoch'] for h in history]
                map_scores = [h['map50'] for h in history]
                ax.plot(epochs, map_scores, 'b-', label='YOLOv10 Fine-tuned', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP@0.5')
            ax.set_title('CARLA - YOLO Models')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Udacity - Original Paper
        ax = axes[1, 0]
        if 'Udacity' in results and 'original_paper' in results['Udacity']:
            history = results['Udacity']['original_paper']['training_history']
            epochs = [h['epoch'] for h in history]
            train_loss = [h['train_loss'] for h in history]
            val_loss = [h['val_loss'] for h in history]
            
            ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Udacity - Original Paper Model')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Udacity - YOLO Models
        ax = axes[1, 1]
        if 'Udacity' in results:
            if 'yolo_testing' in results['Udacity']:
                history = results['Udacity']['yolo_testing']['training_history']
                epochs = [h['epoch'] for h in history]
                map_scores = [h['map50'] for h in history]
                ax.plot(epochs, map_scores, 'g-', label='YOLOv10 Pretrained', linewidth=2, alpha=0.8)
            
            if 'yolo_finetuned' in results['Udacity']:
                history = results['Udacity']['yolo_finetuned']['training_history']
                epochs = [h['epoch'] for h in history]
                map_scores = [h['map50'] for h in history]
                ax.plot(epochs, map_scores, 'b-', label='YOLOv10 Fine-tuned', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP@0.5')
            ax.set_title('Udacity - YOLO Models')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'training_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_comparison_figure(self, results):
        """Create performance comparison figure"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # CARLA - Control Model Metrics
        ax = axes[0, 0]
        if 'CARLA' in results and 'original_paper' in results['CARLA']:
            metrics = results['CARLA']['original_paper']['test_metrics']
            metric_names = ['Steering\nSuccess', 'Throttle\nSuccess', 'Brake\nSuccess']
            metric_values = [metrics['steering_success'], metrics['throttle_success'], metrics['brake_success']]
            
            bars = ax.bar(metric_names, metric_values, color=['#4CAF50', '#2196F3', '#FF9800'])
            ax.set_ylabel('Success Rate')
            ax.set_title('CARLA - Control Model Performance')
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # CARLA - YOLO Metrics
        ax = axes[0, 1]
        if 'CARLA' in results:
            models = []
            precisions = []
            recalls = []
            maps = []
            
            if 'yolo_testing' in results['CARLA']:
                models.append('Pretrained')
                metrics = results['CARLA']['yolo_testing']['evaluation_metrics']
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                maps.append(metrics['map50'])
            
            if 'yolo_finetuned' in results['CARLA']:
                models.append('Fine-tuned')
                metrics = results['CARLA']['yolo_finetuned']['evaluation_metrics']
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                maps.append(metrics['map50'])
            
            x = np.arange(len(models))
            width = 0.25
            
            ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            ax.bar(x, recalls, width, label='Recall', alpha=0.8)
            ax.bar(x + width, maps, width, label='mAP@0.5', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('CARLA - YOLO Model Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        # Udacity - Control Model Metrics
        ax = axes[1, 0]
        if 'Udacity' in results and 'original_paper' in results['Udacity']:
            metrics = results['Udacity']['original_paper']['test_metrics']
            metric_names = ['Steering\nSuccess', 'Throttle\nSuccess', 'Brake\nSuccess']
            metric_values = [metrics['steering_success'], metrics['throttle_success'], metrics['brake_success']]
            
            bars = ax.bar(metric_names, metric_values, color=['#4CAF50', '#2196F3', '#FF9800'])
            ax.set_ylabel('Success Rate')
            ax.set_title('Udacity - Control Model Performance')
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Udacity - YOLO Metrics
        ax = axes[1, 1]
        if 'Udacity' in results:
            models = []
            precisions = []
            recalls = []
            maps = []
            
            if 'yolo_testing' in results['Udacity']:
                models.append('Pretrained')
                metrics = results['Udacity']['yolo_testing']['evaluation_metrics']
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                maps.append(metrics['map50'])
            
            if 'yolo_finetuned' in results['Udacity']:
                models.append('Fine-tuned')
                metrics = results['Udacity']['yolo_finetuned']['evaluation_metrics']
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                maps.append(metrics['map50'])
            
            x = np.arange(len(models))
            width = 0.25
            
            ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
            ax.bar(x, recalls, width, label='Recall', alpha=0.8)
            ax.bar(x + width, maps, width, label='mAP@0.5', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Udacity - YOLO Model Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_ablation_study_figure(self, results):
        """Create comprehensive ablation study figure"""
        if 'ablation' not in results:
            logger.warning("No ablation study results available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ablation Studies - Model Architecture Analysis', fontsize=16, fontweight='bold')
        
        ablation_results = results['ablation']
        
        # Loss comparison
        ax = axes[0, 0]
        models = [f"{r['model_type']}\n{r['feature_dim']}" for r in ablation_results]
        losses = [r['best_val_loss'] for r in ablation_results]
        
        bars = ax.bar(models, losses, alpha=0.8)
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss by Model Configuration')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{loss:.2f}', ha='center', va='bottom', fontsize=8)
        
        # MAE comparison
        ax = axes[0, 1]
        maes = [r['final_metrics']['mae'] for r in ablation_results]
        
        bars = ax.bar(models, maes, alpha=0.8, color='orange')
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Model Configuration')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mae:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Steering success comparison
        ax = axes[0, 2]
        steering_success = [r['final_metrics']['steering_success'] for r in ablation_results]
        
        bars = ax.bar(models, steering_success, alpha=0.8, color='green')
        ax.set_ylabel('Steering Success Rate')
        ax.set_title('Steering Success by Model Configuration')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        for bar, success in zip(bars, steering_success):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{success:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Model type comparison
        ax = axes[1, 0]
        model_types = list(set(r['model_type'] for r in ablation_results))
        avg_losses = []
        
        for mt in model_types:
            mt_results = [r for r in ablation_results if r['model_type'] == mt]
            avg_losses.append(np.mean([r['best_val_loss'] for r in mt_results]))
        
        bars = ax.bar(model_types, avg_losses, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_ylabel('Average Validation Loss')
        ax.set_title('Average Performance by Model Type')
        ax.grid(True, axis='y', alpha=0.3)
        
        for bar, loss in zip(bars, avg_losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{loss:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Feature dimension effect
        ax = axes[1, 1]
        feature_dims = sorted(list(set(r['feature_dim'] for r in ablation_results)))
        simple_avg = []
        conditional_avg = []
        
        for dim in feature_dims:
            simple_results = [r for r in ablation_results if r['model_type'] == 'simple' and r['feature_dim'] == dim]
            conditional_results = [r for r in ablation_results if r['model_type'] == 'conditional' and r['feature_dim'] == dim]
            
            simple_avg.append(np.mean([r['best_val_loss'] for r in simple_results]) if simple_results else 0)
            conditional_avg.append(np.mean([r['best_val_loss'] for r in conditional_results]) if conditional_results else 0)
        
        ax.plot(feature_dims, simple_avg, 'o-', label='Simple', linewidth=2, markersize=8, color='#FF6B6B')
        ax.plot(feature_dims, conditional_avg, 's-', label='Conditional', linewidth=2, markersize=8, color='#4ECDC4')
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Average Validation Loss')
        ax.set_title('Feature Dimension Impact')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance heatmap
        ax = axes[1, 2]
        model_types = ['simple', 'conditional', 'cnn_backbone']
        feature_dims = [256, 512, 1024]
        
        heatmap_data = []
        for mt in model_types:
            row = []
            for fd in feature_dims:
                result = next((r for r in ablation_results if r['model_type'] == mt and r['feature_dim'] == fd), None)
                if result:
                    row.append(result['final_metrics']['steering_success'])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=feature_dims, yticklabels=model_types, ax=ax)
        ax.set_title('Steering Success Heatmap')
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Model Type')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ablation_studies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_table_figure(self, results):
        """Create summary table figure"""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        table_data = []
        headers = ['Dataset', 'Model', 'Epochs', 'Key Metric', 'Performance', 'Status']
        
        for dataset in ['CARLA', 'Udacity']:
            if dataset in results:
                # Original paper model
                if 'original_paper' in results[dataset]:
                    metrics = results[dataset]['original_paper']['test_metrics']
                    table_data.append([
                        dataset,
                        'Original Paper',
                        results[dataset]['original_paper']['training_epochs'],
                        'Steering Success',
                        f"{metrics['steering_success']:.3f}",
                        '✓ Excellent'
                    ])
                
                # YOLO testing
                if 'yolo_testing' in results[dataset]:
                    metrics = results[dataset]['yolo_testing']['evaluation_metrics']
                    table_data.append([
                        dataset,
                        'YOLOv10 Pretrained',
                        results[dataset]['yolo_testing']['training_epochs'],
                        'mAP@0.5',
                        f"{metrics['map50']:.3f}",
                        '⚠ Limited'
                    ])
                
                # YOLO fine-tuned
                if 'yolo_finetuned' in results[dataset]:
                    metrics = results[dataset]['yolo_finetuned']['evaluation_metrics']
                    table_data.append([
                        dataset,
                        'YOLOv10 Fine-tuned',
                        results[dataset]['yolo_finetuned']['training_epochs'],
                        'mAP@0.5',
                        f"{metrics['map50']:.3f}",
                        '✓ Excellent'
                    ])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colColours=['#f2f2f2']*6)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code performance
        for i in range(1, len(table_data) + 1):
            status = table_data[i-1][5]
            if '✓' in status:
                table[(i, 5)].set_facecolor('#E8F5E8')
            elif '⚠' in status:
                table[(i, 5)].set_facecolor('#FFF3E0')
        
        plt.title('Experiment Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.figures_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_dataset_comparison_figure(self, results):
        """Create dataset comparison figure"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('CARLA vs Udacity Dataset Comparison', fontsize=16, fontweight='bold')
        
        # Dataset sizes
        ax = axes[0]
        datasets = ['CARLA', 'Udacity']
        sizes = [50000, 8036]
        
        bars = ax.bar(datasets, sizes, alpha=0.8, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset Size Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                   f'{size:,}', ha='center', va='bottom', fontweight='bold')
        
        # Control model comparison
        ax = axes[1]
        if 'CARLA' in results and 'Udacity' in results:
            if 'original_paper' in results['CARLA'] and 'original_paper' in results['Udacity']:
                carla_metrics = results['CARLA']['original_paper']['test_metrics']
                udacity_metrics = results['Udacity']['original_paper']['test_metrics']
                
                metrics = ['Steering\nSuccess', 'Throttle\nSuccess', 'Brake\nSuccess']
                carla_values = [carla_metrics['steering_success'], carla_metrics['throttle_success'], carla_metrics['brake_success']]
                udacity_values = [udacity_metrics['steering_success'], udacity_metrics['throttle_success'], udacity_metrics['brake_success']]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax.bar(x - width/2, carla_values, width, label='CARLA', alpha=0.8, color='#FF6B6B')
                ax.bar(x + width/2, udacity_values, width, label='Udacity', alpha=0.8, color='#4ECDC4')
                
                ax.set_xlabel('Control Metric')
                ax.set_ylabel('Success Rate')
                ax.set_title('Control Model Performance')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics)
                ax.legend()
                ax.set_ylim(0, 1)
                ax.grid(True, axis='y', alpha=0.3)
        
        # YOLO model comparison
        ax = axes[2]
        models = ['Pretrained', 'Fine-tuned']
        carla_maps = []
        udacity_maps = []
        
        if 'CARLA' in results:
            if 'yolo_testing' in results['CARLA']:
                carla_maps.append(results['CARLA']['yolo_testing']['evaluation_metrics']['map50'])
            else:
                carla_maps.append(0)
            if 'yolo_finetuned' in results['CARLA']:
                carla_maps.append(results['CARLA']['yolo_finetuned']['evaluation_metrics']['map50'])
            else:
                carla_maps.append(0)
        
        if 'Udacity' in results:
            if 'yolo_testing' in results['Udacity']:
                udacity_maps.append(results['Udacity']['yolo_testing']['evaluation_metrics']['map50'])
            else:
                udacity_maps.append(0)
            if 'yolo_finetuned' in results['Udacity']:
                udacity_maps.append(results['Udacity']['yolo_finetuned']['evaluation_metrics']['map50'])
            else:
                udacity_maps.append(0)
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, carla_maps, width, label='CARLA', alpha=0.8, color='#FF6B6B')
        ax.bar(x + width/2, udacity_maps, width, label='Udacity', alpha=0.8, color='#4ECDC4')
        
        ax.set_xlabel('YOLO Model')
        ax.set_ylabel('mAP@0.5')
        ax.set_title('YOLO Model Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_figures(self):
        """Generate all result figures"""
        logger.info("Loading results for figure generation...")
        results = self.load_all_results()
        
        logger.info("Generating training progression figure...")
        self.create_training_progression_figure(results)
        
        logger.info("Generating performance comparison figure...")
        self.create_performance_comparison_figure(results)
        
        logger.info("Generating ablation study figure...")
        self.create_ablation_study_figure(results)
        
        logger.info("Generating summary table figure...")
        self.create_summary_table_figure(results)
        
        logger.info("Generating dataset comparison figure...")
        self.create_dataset_comparison_figure(results)
        
        logger.info(f"All figures generated and saved to {self.figures_dir}")
        
        # Create figure index
        figure_index = f"""
# Result Figures Index

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Available Figures

1. **training_progression.png** - Training curves for all models across epochs
2. **performance_comparison.png** - Final performance metrics comparison
3. **ablation_studies.png** - Comprehensive ablation study analysis
4. **summary_table.png** - Summary table of all experiments
5. **dataset_comparison.png** - CARLA vs Udacity dataset comparison

## Figure Descriptions

### Training Progression
Shows loss curves and mAP progression for all models during training, demonstrating convergence patterns and learning dynamics.

### Performance Comparison
Compares final performance metrics across different models and datasets, highlighting relative strengths and weaknesses.

### Ablation Studies
Analyzes the impact of different architectural choices including model types, feature dimensions, and conditioning strategies.

### Summary Table
Provides a comprehensive overview of all experiments with key metrics and performance indicators.

### Dataset Comparison
Direct comparison between CARLA (synthetic) and Udacity (real-world) datasets across different model types.

## Usage Guidelines

All figures are generated at 300 DPI for publication quality.
Figures use consistent color schemes and styling for professional presentation.
"""
        
        with open(self.figures_dir / 'README.md', 'w') as f:
            f.write(figure_index)
        
        logger.info("Figure index created")

def main():
    """Main function"""
    generator = ResultFiguresGenerator()
    generator.generate_all_figures()

if __name__ == "__main__":
    main()
