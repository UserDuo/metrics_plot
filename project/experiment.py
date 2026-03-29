import pandas as pd
import numpy as np
from typing import List, Dict, Any
from project.skills.registry import registry
from project.skills.metric_skills import register_all_metrics
from scipy.stats import bootstrap

class ExperimentRunner:
    def __init__(self):
        register_all_metrics()
        
    def run_study(self, data: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Runs the full metric suite on a list of sentence pairs.
        data: [{'s': '...', 'sp': '...'}, ...]
        """
        results = []
        for i, pair in enumerate(data):
            s, sp = pair['s'], pair['sp']
            try:
                # We reuse the composite score skill but we also want raw values
                # Ideally we should call skills individually or update the composite skill to return everything
                # For now, let's use the composite skill which returns 'details'
                res = registry.execute("calculate_composite_score", s=s, sp=sp)
                
                row = {
                    'id': i,
                    'original': s,
                    'disturbed': sp,
                    'composite_score': res['composite_score']
                }
                # Flatten details
                for cat, val in res['details'].items():
                    row[f'score_{cat}'] = val
                    
                results.append(row)
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                
        return pd.DataFrame(results)

    def analyze_results(self, df: pd.DataFrame):
        """
        Performs statistical analysis on the results.
        Calculates Mean, Std, and 95% Confidence Intervals via Bootstrapping.
        """
        stats = []
        score_cols = [c for c in df.columns if c.startswith('score_') or c == 'composite_score']
        
        for col in score_cols:
            data = df[col].values
            mean = np.mean(data)
            std = np.std(data)
            
            # Bootstrap CI
            if len(data) > 1:
                res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000)
                ci_low = res.confidence_interval.low
                ci_high = res.confidence_interval.high
            else:
                ci_low = mean
                ci_high = mean
            
            stats.append({
                'Metric': col,
                'Mean': mean,
                'Std': std,
                'CI_95_Low': ci_low,
                'CI_95_High': ci_high
            })
            
        return pd.DataFrame(stats)
