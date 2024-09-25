
# GPT-2 from Scratch

## Overview

This project involves building a **GPT-2 model** from scratch. It follows the principles of the **Transformer architecture** and focuses on creating a custom implementation of the GPT-2 model. The project covers every stage, allowing you to fully understand how GPT-2 works at a low level.

## Features

- Implementation of **GPT-2 architecture** from scratch.
- Customizable **hyperparameters** for  fine-tuning.
- Support for **GPU**-based training for large-scale models.
- Modular and extensible code for experimenting with different layers and configurations.
- **Text generation** based on pre-trained or custom-trained models.

## Project Structure

```
├── data/                   # Directory for dataset storage
├── checkpoints/            # Directory for GPT-2 checkpoints
├── src/                    # Source code for the project
│   ├── model/              # Contains the GPT-2 model code
│   ├── train.py            # Script for training the model
│   ├── infer.py            # Script for inference/generation
│   ├── tokenizer.py        # Tokenization logic
│   ├── utils.py            # Utility functions like data preprocessing
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gpt2-from-scratch.git
   cd gpt2-from-scratch
   ```

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download any pre-trained GPT-2 checkpoints (if available) and place them in the `checkpoints/` directory.



## Inference

To generate text based on a pre-trained or custom-trained model, you can run:

```bash
python src/infer.py--checkpoint checkpoints/gpt2_checkpoint.pt
```

This will generate text based on the input prompt using the specified model checkpoint.

## Checkpoints

The model saves checkpoints periodically during training, which can be loaded later for inference or to continue training. You can load checkpoints from the `checkpoints/` folder for both inference and fine-tuning.

## Customization

You can customize the GPT-2 model architecture, the training loop, or the tokenization mechanism by modifying the code in the `src/` folder. The project is designed to be modular, so you can experiment with various layers, configurations, and training strategies.

## Acknowledgments

- Inspired by **OpenAI's GPT-2** model.
- Utilizes concepts from the **Transformer architecture** and **Hugging Face** tokenizers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

