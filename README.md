# appl-embedding-visualizer
Interactive visualization of text embeddings

![tsne-2](https://github.com/pbl-nl/appl-embedding-visualizer/assets/7226328/bd0bef83-692c-43e1-a5a1-7117894b85ff)

## How to use this repo
! This repo is tested on a Windows platform

### Preparation
1. Clone this repo to a folder of your choice
2. In the root folder, create a file named ".env" and enter your OpenAI API key in the first line of this file in case you want to use the OpenAI API:<br>
OPENAI_API_KEY="sk-....."<br>
3. Save and close the .env file<br>
* If you don't have an OpenAI API key yet, you can obtain one here: https://platform.openai.com/account/api-keys
* Click on + Create new secret key
* Enter an identifier name (optional) and click on Create secret key
4. In case you want to use one of the open source models API's that are available on huggingface:<br>
Enter your Hugging Face API key in the ".env" file :<br>
HUGGINGFACEHUB_API_TOKEN="hf_....."<br>
* If you don't have an Hugging Face API key yet, you can register at https://huggingface.co/join
* When registered and logged in, you can get your API key in your Hugging Face profile settings
  
### Conda virtual environment setup
1. Open an Anaconda prompt or other command prompt
2. Go to the root folder of the project and create a Python environment with conda with <code>conda env create -f appl-embedding-visualizer.yml</code><br>
NB: The name of the environment is appl-embedding-visualizer by default. It can be changed to a name of your choice in the first line of the file appl-embedding-visualizer.yml
3. Activate this environment with <code>conda activate appl-embedding-visualizer</code>

### Run the application
within the activated conda environment, type <code>python tsne.py</code>
