import matplotlib.pyplot as plt

def plot_auroc_score(results, model_name='GPT-2 Large', lens_type='Tuned Lens'):
    # Results data
    methods = ['Isolation Forest', 'LOF', 'SRM Baseline']
    scores = [results['iforest_auroc'], results['lof_auroc'], results['srm_auroc']]
    colors = ['blue', 'green', 'orange']  # Custom bar colors

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, scores, color=colors, alpha=0.8)

    # Add title and labels
    plt.title('Prompt Injection Detection Performance ({model_name}) with {lens_type}', fontsize=14)
    plt.ylabel('AUROC Score', fontsize=12)
    plt.xlabel('Detection Methods', fontsize=12)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add annotations above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=10)

    # Add horizontal line at AUROC=0.5 for reference (random guessing)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Baseline (AUROC = 0.5)')

    # Limit y-axis to [0, 1]
    plt.ylim(0, 1)

    # Add legend
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()