# magic the gathering card generation script

this is a natural language processing project to train gpt-2 model on Magic the Gathering cards and generate new cards from a prompt.
the goal was to see how accurately trained models could recreate Magic the Gathering cards, and how creative it could be with new cards.

# how to run/usage 

for this project you will need to download the AtomicCards database of magic cards from this website: https://mtgjson.com/downloads/all-files/
  i cannot include this database due to GitHub file size limits (i tried uploading from console it didnt work ;-;)
you will also need to install related packages i.e. torch, transformers, dataset etc
then encode the database for training by running encoder.py
  you may need to change file paths at input and output path at the top of the script
run model.py and change the model name to whatever you want at the top in MODEL_OUT
  again you may need to change file paths in the script to where your data is stored
  feel free to adjust hyperparameters to your liking, i.e. full database, more epochs, different training and tokenizing model etc
finally run script.py with your new model that you saved with MODEL_OUT
  change generate_card() with desired prompt
