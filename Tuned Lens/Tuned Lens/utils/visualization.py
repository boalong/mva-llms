import matplotlib.pyplot as plt
import seaborn as sns

def plot_kl_perplexity(avg_klavg_ppl_tuned, avg_kl_tuned, avg_ppl_logit, avg_kl_logit):
    layers = list(avg_kl_tuned.keys())

    # Enhanced KL Divergence and Perplexity plots
    plt.figure(figsize=(14, 6))

    # KL Divergence Plot
    plt.subplot(1, 2, 1)
    plt.plot(layers, avg_kl_tuned.values(), label='Tuned Lens', marker='x', linestyle='--', color='blue')
    plt.plot(layers, avg_kl_logit.values(), label='Logit Lens', marker='o', linestyle='-', color='orange')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("KL Divergence", fontsize=12)
    plt.title("KL Divergence Across Layers", fontsize=14)
    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(fontsize=10)

    # Add annotations for key points (example: first and last layers)
    plt.text(layers[0], avg_kl_tuned[layers[0]], f'{avg_kl_tuned[layers[0]]:.2f}', ha='center', va='bottom', fontsize=10, color='blue')
    plt.text(layers[-1], avg_kl_logit[layers[-1]], f'{avg_kl_logit[layers[-1]]:.2f}', ha='center', va='bottom', fontsize=10, color='orange')

    # Perplexity Plot
    plt.subplot(1, 2, 2)
    plt.plot(layers, avg_ppl_logit.values(), label='Logit Lens', marker='o', linestyle='-', color='green')
    plt.plot(layers, avg_ppl_tuned.values(), label='Tuned Lens', marker='x', linestyle='--', color='purple')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Perplexity Across Layers", fontsize=14)
    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(fontsize=10)

    # Add annotations for key points (example: first and last layers)
    plt.text(layers[0], avg_ppl_tuned[layers[0]], f'{avg_ppl_tuned[layers[0]]:.2f}', ha='center', va='bottom', fontsize=10, color='purple')
    plt.text(layers[-1], avg_ppl_logit[layers[-1]], f'{avg_ppl_logit[layers[-1]]:.2f}', ha='center', va='bottom', fontsize=10, color='green')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display plot
    plt.show()