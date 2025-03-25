# MVA Project: The Tuned Lens

This is the project of the Group 25 for the LLM course at the MVA. The authors are:
* Omar Arbi
* Mouad Id Sougou
* Adrien Letellier
* Jad Zakharia

In this repository you will find a reproduction of the Tuned Lens method described in the article [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112), and some applications.

## Structure of the repository

In the `reproducing` folder, you will find:
* modules to reproduce the method of the article on the trivial addition task in the `with_classes` folder.
* a notebook that does the same in a self-contained way in the `self_contained` folder.
* a `Tuned Lens` folder which reproduces the method on a GPT-2 model. More detailed instructions are available in the readme.

In the `experiments` folder, you will find:
* a `Llama_thinks_eng.ipynb` notebook appying the tuned lens to the translation setting to see if the model uses an English internal representation, as in the article [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://aclanthology.org/2024.acl-long.820/). 
* a `yes_no_llama3.ipynb` notebook that attempts to see if the model uses intermediate layers to have a rough representation of what it will output and the last layers to format it, by playing the game "Yes, no, black, white" ("Ni oui, ni non" in French) with the LLM.
To run this notebook in the `experiments` folder, you can install our debugged version of the `tuned-lens` library. First, clone the repository using:
`git clone https://github.com/boalong/mva-llms-tuned-lens.git`
Then, from the root of this cloned repository, do:
`pip install .`
* a `Prompt injection` folder, in which you will find a README detailing the instructions for this experiment. In this folder, we attempted to reproduce and extend some experiments of the original article to detect prompt injection.