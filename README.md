# Legaltensor Subnet

Legaltensor is a Bittensor subnet focused on addressing legal questions and providing assistance in the legal domain. By combining state-of-the-art AI models with a collaborative network, Legaltensor aims to deliver reliable legal insights to users.

---

## 🌟 Key Features

- **Legal Expertise**: Tailored for answering legal questions and supporting users with legal challenges.
- **Collaborative Network**:  
  - **Miners**: Solve legal tasks using advanced AI models.  
  - **Validators**: Ensure the accuracy and relevance of solutions provided by miners.
- **Advanced AI Models**: Powered by the [Saul-7B-Base](https://huggingface.co/Equall/Saul-7B-Base) model, specialized in legal tasks.
- **Comprehensive Dataset**: Built on the [LegalBench dataset](https://github.com/HazyResearch/legalbench), offering diverse legal challenges and benchmarks.

---

## 🛠️ Architecture Overview

Legaltensor operates as a Bittensor subnet with a miner-validator structure:

1. **Validators**  
   - Query miners with legal tasks.  
   - Validate the quality and reliability of responses.  
   - Store validated responses in the distributed system.

2. **Miners**  
   - Process tasks using the Saul-7B-Base model.  
   - Respond to queries from validators with high-quality outputs.  
   - Utilize the LegalBench dataset for task-specific benchmarks.

---

## 🚀 Getting Started

### Quick Start

#### Prerequisites

1. Clone the repository:
   ```bash
   git clone https://github.com/LegalTensor/legaltensor.git
   cd legaltensor
   ```

2. Init the submodule:
   ```bash
   git submodule update --init
   ```

3. Environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install Dependencies:
   ```bash
   pip3 install -r requirements.txt
   export PYTHONPATH=.
   ```

#### Run miner
   ```bash
   python neurons/miner/miner.py --subtensor.network test --testnet 205 --wallet.name [wallet name] --wallet.hotkey [wallet.hotkey]
   ```

#### Run validator
   ```bash
   python neurons/validator/validator.py --subtensor.network test --testnet 205 --wallet.name [wallet name] --wallet.hotkey [wallet.hotkey]
   ```

### Run with pm2

#### Run miner using pm2
   ```bash
   pm2 start python --name legaltensor-miner -- neurons/miner/miner.py --subtensor.network test --testnet 205 --wallet.name [wallet name] --wallet.hotkey [wallet.hotkey]
   ```

#### Run validator using pm2
   ```bash
   pm2 start python --name legaltensor-validator -- neurons/validator/validator.py --subtensor.network test --testnet 205 --wallet.name [wallet name] --wallet.hotkey [wallet.hotkey]
   ```

---

## 📊 Dataset

Legaltensor leverages the [LegalBench dataset](https://github.com/HazyResearch/legalbench), a rich resource of tasks tailored for the legal field. This dataset enables miners to address diverse legal challenges effectively.

---

## 🤝 Contribution Guide

We welcome contributions from the community. You can help improve Legaltensor by:

1. Forking the repository.
2. Creating a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Committing your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Submitting a pull request.

---

## 🔗 Resources

- [Bittensor Documentation](https://docs.bittensor.com/)
- [Saul-7B-Base Model](https://huggingface.co/Equall/Saul-7B-Base)
- [LegalBench Dataset](https://github.com/HazyResearch/legalbench)

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
