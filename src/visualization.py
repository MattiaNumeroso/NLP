"""Modulo per visualizzazione dei risultati."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(importance_df, title="Feature Importance", save_path=None):
    """Barplot delle features più importanti."""
    plt.figure(figsize=(10, 8))

    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')

    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
