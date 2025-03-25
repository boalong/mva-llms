from datasets import load_dataset
import random



class PromptDataset:
    """
    A dataset class for handling normal and adversarial prompts.
    """
    def __init__(
            self, 
            normal_data: list, 
            adversarial_data: list):
        """
        Initializes the dataset with normal and adversarial prompts.
        
        Parameters:
        normal_data (list): A list of normal prompt texts.
        adversarial_data (list): A list of adversarial prompt texts.
        """
        self.normal_prompts = normal_data
        self.adversarial_prompts = adversarial_data

    @classmethod
    def from_huggingface(cls, normal_dataset: str = 'wikitext',
                         adversarial_dataset: str = 'rubend18/ChatGPT-Jailbreak-Prompts'):
        """
        Loads datasets from Hugging Face.
        
        Parameters:
        normal_dataset (str): The dataset name for normal prompts.
        adversarial_dataset (str): The dataset name for adversarial prompts.
        
        Returns:
        PromptDataset: An instance containing the loaded datasets.
        """
        normal = load_dataset(normal_dataset, 'wikitext-103-v1', split='train[:300]')
        adversarial = load_dataset(adversarial_dataset, split='train[:300]')
        return cls(
            [t for t in normal['text'] if t.strip()],
            [p['Prompt'] for p in adversarial if p['Prompt'].strip()]
        )

    def split(self, train_ratio: float = 0.8):
        """
        Splits the dataset into training and testing sets, ensuring all adversarial prompts are in the test set.
        
        Parameters:
        train_ratio (float): The proportion of normal prompts to include in the training set.
        
        Returns:
        tuple: Two PromptDataset instances (train and test datasets).
        """
        normal_data = [(p, 0) for p in self.normal_prompts]
        adversarial_data = [(p, 1) for p in self.adversarial_prompts]
        random.shuffle(normal_data)
        split_idx = int(len(normal_data) * train_ratio)
        train_normal = normal_data[:split_idx]
        test_normal = normal_data[split_idx:]
        test_adversarial = adversarial_data[:len(test_normal)]
        return (
            PromptDataset([p for p, _ in train_normal], []),
            PromptDataset([p for p, _ in test_normal], [p for p, _ in test_adversarial])
        )
