# magic the gathering card generation script

this is a natural language processing project to train a gpt-2 model on Magic the Gathering cards and generate new cards from a prompt.<br/>
the goal was to see how accurately trained models could recreate Magic the Gathering cards, and how creative it could be with new cards.<br/>

## how to run/usage 

### data preperation
- for this project you will need to download the AtomicCards database of magic cards from this website: https://mtgjson.com/downloads/all-files/<br/>
    - i cannot include this database due to GitHub file size limits (i tried uploading from console it didnt work ;-;)<br/>
- you will also need to install related packages i.e. torch, transformers, dataset etc<br/>
### encoding
- then encode the database for training by running encoder.py<br/>
    - you may need to change file paths at input and output path at the top of the script<br/>
### training model
- run model.py and change the model name to whatever you want at the top in MODEL_OUT<br/>
    - again you may need to change file paths in the script to where your data is stored<br/>
    - feel free to adjust hyperparameters to your liking, i.e. full database, more epochs, different training and tokenizing model etc<br/>
### running model
- finally run script.py with your new model that you saved with MODEL_OUT<br/>
    - change generate_card() with desired prompt<br/>

##Technology Stack

- Python
- PyTorch – deep learning framework
- Hugging Face Transformers – GPT-2 model and tokenization
- Hugging Face Datasets – dataset loading and preprocessing
- CUDA (optional) – GPU acceleration when available

## author

jacob bloom
https://www.linkedin.com/in/jacobmbloom/
