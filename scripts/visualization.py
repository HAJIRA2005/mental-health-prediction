"""
Visualization utilities for mental health prediction project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_data_distribution(df, output_path=None):
    """Plot feature and target distribution"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Mental Health Dataset Distribution', fontsize=16, fontweight='bold')
    
    columns = df.columns[:-1]  # Exclude target
    
    for idx, col in enumerate(columns):
        ax = axes[idx // 3, idx % 3]
        ax.hist(df[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[2, 2])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
    plt.show()

def plot_class_distribution(df, output_path=None):
    """Plot class distribution"""
    plt.figure(figsize=(10, 6))
    class_dist = df['mental_health'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = plt.bar(class_dist.index, class_dist.values, color=colors, edgecolor='black', alpha=0.8)
    
    plt.title('Mental Health Status Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Mental Health Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {output_path}")
    plt.show()

def plot_correlation_matrix(df, output_path=None):
    """Plot correlation matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Encode target for correlation
    df_corr = df.copy()
    df_corr['mental_health'] = (df_corr['mental_health'] == 'Good').astype(int)
    
    corr_matrix = df_corr.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix plot saved to {output_path}")
    plt.show()

def main():
    data_path = Path(__file__).parent.parent / "data" / "mental_health_data.csv"
    
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print("Please run data_generator.py first")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot class distribution
    plot_class_distribution(
        df,
        output_path=Path(__file__).parent.parent / "models" / "class_distribution.png"
    )
    
    # Plot feature distribution
    plot_data_distribution(
        df,
        output_path=Path(__file__).parent.parent / "models" / "feature_distribution.png"
    )
    
    # Plot correlation matrix
    plot_correlation_matrix(
        df,
        output_path=Path(__file__).parent.parent / "models" / "correlation_matrix.png"
    )
    
    print("Visualizations completed!")

if __name__ == "__main__":
    main()
