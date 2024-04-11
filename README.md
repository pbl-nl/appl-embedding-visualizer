# appl-embedding-visualizer
Interactive visualization of text embeddings

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
