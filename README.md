# DaviRAG

## Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

## Setup

1. **Create a Conda Environment**

   Create a new Conda environment named `langchain_env` with Python 3.10 (or your preferred version):

   ```bash
   conda create -n langchain_env python=3.10
   ```
2. **Activate the Environment**
   
   Activate the `langchain_env` environment:
   ```bash
   conda activate langchain_env
   ```

3. **Install Dependencies**

   Install the required dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Create the Vector Database**
   
   Run the `create_vector_db.py` script to generate the vector database:

   ```bash
   python create_vector_db.py
   ```

2. **Run the Testing Script**
   
   After creating the vector database, you can run the testing.py script to test the functionality:

   ```bash
   streamlit run testing.py
   ```   
