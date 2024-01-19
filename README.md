# MarkerTranslator

Getting started:

There are two entry points:

- `python3 main.py train` for training a model
- `python3 main.py visualize` for running a 3D visualizer in the browser to see what your model is doing

## Training

You'll need to get the data from here, and put it in `data/` in this repo: https://drive.google.com/drive/u/1/folders/1mdwQnVh2-4-hOtoMFrzEe5ABP2UXQZ_I

Then to verify everything is kinda working, you can run `python3 main.py train --overfit --no-wandb` to attempt a very quick overfitting run on a single batch of data. This will not log to Weights and Biases.

If that doesn't crash, then you're good to train the model properly. Start with `python3 main.py train`, and then you can use the Weights and Biases dashboard to monitor the training progress.
From there, you can experiment with the command line flags to train different layer sizes, different depths of models, different optimizers, different learning rates, etc.

## Visualizing

Once you've got a trained model, you can run `python3 main.py visualize` to compare the original and reconstructed poses in a 3D visualizer in the browser. This will start a server on port 8080, so you can go to http://localhost:8080 to see the visualizer.
The reconstructed model is rendered in red, original in white.