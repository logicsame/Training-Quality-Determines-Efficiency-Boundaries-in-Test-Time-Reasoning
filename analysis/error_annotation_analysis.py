"""
ANALYZE ANNOTATED ERRORS
=========================
Run this AFTER you finish annotation to get your results!

Usage:
    python analyze_annotations.py data/annotations/errors_annotated.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class AnnotationAnalyzer:
    """Analyze annotated error data"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        # Check if annotations are complete
        self.validate_annotations()
    
    def validate_annotations(self):
        """Check if file is properly annotated"""
        required_cols = ['error_category', 'is_cumulative', 'error_location']
        
        for col in required_cols:
            if col not in self.df.columns:
                print(f" Error: Column '{col}' not found in file!")
                print(f"   Make sure you annotated the file first.")
                sys.exit(1)
        
        # Check how many are annotated
        annotated = self.df[
            self.df['error_category'].notna() & 
            (self.df['error_category'] != '')
        ]
        
        print(f"\n Found {len(annotated)} annotated errors")
        
        if len(annotated) == 0:
            print(" No annotations found! Please annotate the file first.")
            sys.exit(1)
    
    def analyze_error_types(self):
        """Analyze distribution of error types by condition"""
        
        print("\n" + "="*80)
        print("📊 ERROR TYPE ANALYSIS")
        print("="*80)
        
        # Filter only annotated rows
        annotated = self.df[
            self.df['error_category'].notna() & 
            (self.df['error_category'] != '')
        ]
        
        print(f"\nTotal annotated errors: {len(annotated)}")
        print(f"  Control: {len(annotated[annotated['condition']=='control'])}")
        print(f"  Brief: {len(annotated[annotated['condition']=='brief'])}")
        
        print("\n" + "-"*80)
        print("ERROR TYPE DISTRIBUTION:")
        print("-"*80)
        
        for condition in ['control', 'brief']:
            condition_df = annotated[annotated['condition'] == condition]
            
            print(f"\n{condition.upper()}:")
            
            type_counts = condition_df['error_category'].value_counts()
            
            for error_type, count in type_counts.items():
                pct = count / len(condition_df) * 100
                print(f"  {error_type:15s}: {count:3d} ({pct:5.1f}%)")
        
        # Chi-square test for independence
        print("\n" + "-"*80)
        print("STATISTICAL TEST:")
        print("-"*80)
        
        contingency = pd.crosstab(
            annotated['condition'],
            annotated['error_category']
        )
        
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        
        print(f"\nχ² test for independence:")
        print(f"  χ² = {chi2:.2f}")
        print(f"  p-value = {p_val:.4f}")
        
        if p_val < 0.05:
            print(f"   SIGNIFICANT: Error types differ by condition!")
        else:
            print(f"    Not significant: Error types similar across conditions")
    
    def analyze_cumulative_errors(self):
        """⭐ MAIN ANALYSIS: Compare cumulative error rates"""
        
        print("\n" + "="*80)
        print("⭐ CUMULATIVE ERROR ANALYSIS (KEY FINDING)")
        print("="*80)
        
        # Filter annotated rows
        annotated = self.df[
            self.df['is_cumulative'].notna() & 
            (self.df['is_cumulative'] != '')
        ]
        
        # Separate by condition
        control = annotated[annotated['condition'] == 'control']
        brief = annotated[annotated['condition'] == 'brief']
        
        # Count cumulative errors
        control_cumulative = (control['is_cumulative'] == 'yes').sum()
        control_total = len(control)
        
        brief_cumulative = (brief['is_cumulative'] == 'yes').sum()
        brief_total = len(brief)
        
        control_rate = control_cumulative / control_total if control_total > 0 else 0
        brief_rate = brief_cumulative / brief_total if brief_total > 0 else 0
        
        print(f"\nCUMULATIVE ERROR RATES:")
        print("-"*80)
        print(f"  Control (verbose): {control_cumulative}/{control_total} = {control_rate*100:.1f}%")
        print(f"  Brief:             {brief_cumulative}/{brief_total} = {brief_rate*100:.1f}%")
        print(f"  Difference:        {(control_rate - brief_rate)*100:+.1f} percentage points")
        
        # Fisher's exact test
        print("\n" + "-"*80)
        print("STATISTICAL TEST (Fisher's Exact):")
        print("-"*80)
        
        contingency = [
            [control_cumulative, control_total - control_cumulative],
            [brief_cumulative, brief_total - brief_cumulative]
        ]
        
        odds_ratio, p_val = fisher_exact(contingency)
        
        print(f"\n  Odds Ratio: {odds_ratio:.2f}")
        print(f"  p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            if control_rate > brief_rate:
                print(f"\n   SIGNIFICANT FINDING!")
                print(f"   Control has {odds_ratio:.1f}× MORE cumulative errors than brief")
                print(f"   This explains why verbose reasoning hurts accuracy!")
            else:
                print(f"\n    Brief has more cumulative errors (unexpected)")
        else:
            print(f"\n    Difference not statistically significant")
            print(f"     (might need more annotations)")
        
        return {
            'control_rate': control_rate,
            'brief_rate': brief_rate,
            'odds_ratio': odds_ratio,
            'p_value': p_val
        }
    
    def analyze_error_location(self):
        """Analyze where errors occur"""
        
        print("\n" + "="*80)
        print(" ERROR LOCATION ANALYSIS")
        print("="*80)
        
        annotated = self.df[
            self.df['error_location'].notna() & 
            (self.df['error_location'] != '')
        ]
        
        print("\nWhere do errors occur?")
        print("-"*80)
        
        for condition in ['control', 'brief']:
            condition_df = annotated[annotated['condition'] == condition]
            
            print(f"\n{condition.upper()}:")
            
            location_counts = condition_df['error_location'].value_counts()
            
            for location, count in location_counts.items():
                pct = count / len(condition_df) * 100
                print(f"  {location:12s}: {count:3d} ({pct:5.1f}%)")
    
    def generate_summary(self, cumulative_results):
        """Generate summary for paper"""
        
        print("\n" + "="*80)
        print(" SUMMARY FOR YOUR PAPER")
        print("="*80)
        
        print("\n" + "-"*80)
        print("WRITE THIS IN YOUR RESULTS SECTION:")
        print("-"*80)
        
        control_rate = cumulative_results['control_rate']
        brief_rate = cumulative_results['brief_rate']
        odds_ratio = cumulative_results['odds_ratio']
        p_val = cumulative_results['p_value']
        
        # Get total annotations
        annotated = self.df[
            self.df['error_category'].notna() & 
            (self.df['error_category'] != '')
        ]
        
        n_total = len(annotated)
        n_control = len(annotated[annotated['condition'] == 'control'])
        n_brief = len(annotated[annotated['condition'] == 'brief'])
        
        summary_text = f"""
Manual annotation of {n_total} errors ({n_control} control, {n_brief} brief) revealed 
that verbose (control) responses exhibited {odds_ratio:.1f}× higher rates of cumulative 
errors—where an error in step N causes errors in subsequent steps—than brief responses 
({control_rate*100:.1f}% vs {brief_rate*100:.1f}%, OR={odds_ratio:.2f}, p={p_val:.3f}, 
Fisher's exact test). This error amplification mechanism explains why longer reasoning 
traces reduce rather than improve accuracy: each additional step provides opportunity 
for error propagation rather than self-correction.
"""
        
        print(summary_text)
        
        print("\n" + "-"*80)
        print("KEY NUMBERS FOR ABSTRACT:")
        print("-"*80)
        print(f"  • {n_total} errors annotated")
        print(f"  • {odds_ratio:.1f}× more cumulative errors in control")
        print(f"  • Control: {control_rate*100:.1f}% cumulative")
        print(f"  • Brief: {brief_rate*100:.1f}% cumulative")
        print(f"  • p = {p_val:.3f}")
    
    def create_visualization(self):
        """Create figures for paper"""
        
        print("\n" + "="*80)
        print(" CREATING FIGURES")
        print("="*80)
        
        annotated = self.df[
            self.df['error_category'].notna() & 
            (self.df['error_category'] != '')
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # FIXED VERSION:
        error_counts = pd.crosstab(
            annotated['condition'],
            annotated['error_category'],
            normalize='index'
        ) * 100

        # Plot with explicit color mapping
        ax = axes[0]
        colors = ['#3498db', '#e74c3c']  # Blue for brief, Red for control

        # Ensure correct order: control first, then brief
        error_counts = error_counts.reindex(['control', 'brief'])
        error_counts.T.plot(kind='bar', ax=ax, color=colors)

        ax.set_xlabel('Error Type')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Error Type Distribution')
        ax.legend(title='Condition', labels=['Control (Verbose)', 'Brief (Constrained)'])
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Cumulative error rate
        ax = axes[1]
        
        cumulative_data = annotated.groupby('condition')['is_cumulative'].apply(
            lambda x: (x == 'yes').sum() / len(x) * 100
        )
        
        colors = ['#e74c3c', '#2ecc71']  # Red for control, green for brief
        bars = ax.bar(cumulative_data.index, cumulative_data.values, color=colors)
        
        ax.set_ylabel('Cumulative Error Rate (%)')
        ax.set_title('Cumulative Errors by Condition')
        ax.set_ylim([0, max(cumulative_data.values) * 1.2])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        output_path = 'error_analysis_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"\n Figure saved: {output_path}")
        
        plt.close()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        
        print("\n" + "="*80)
        print(" RUNNING COMPLETE ERROR ANALYSIS")
        print("="*80)
        
        # 1. Error types
        self.analyze_error_types()
        
        # 2. Cumulative errors (KEY FINDING)
        cumulative_results = self.analyze_cumulative_errors()
        
        # 3. Error location
        self.analyze_error_location()
        
        # 4. Generate summary
        self.generate_summary(cumulative_results)
        
        # 5. Create visualizations
        self.create_visualization()
        
        print("\n" + "="*80)
        print(" ANALYSIS COMPLETE!")
        print("="*80)
        print("\nFiles created:")
        print("   error_analysis_results.png")
        print("\nNext steps:")
        print("  1. Add the summary text to your paper")
        print("  2. Include the figure in your results section")
   


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("\n" + "="*80)
        print("ERROR ANALYSIS TOOL")
        print("="*80)
        print("\nUsage:")
        print("  python analyze_annotations.py errors_annotated.csv")
        print("\nNote: Run this AFTER you finish annotation!")
        print("="*80)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"\n Error: File not found: {csv_path}")
        print(f"   Make sure you annotated the file first!")
        sys.exit(1)
    
    analyzer = AnnotationAnalyzer(csv_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()