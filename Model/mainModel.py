"""Entry point for training the fingertip force estimator end to end (see
force_estimation build prompt, Section 3). Loads config.yaml's
model_training block, hands off to dataset.py/model.py/train.py to build
the datasets, build the model, and run the full training loop, then
reports where the checkpoint and metrics history ended up.

Run with the trajPipeline conda env:

    /home/wouterdg/anaconda3/envs/trajPipeline/bin/python mainModel.py
"""

import time                                                                    #For timing the whole run

import torch                                                                   #For picking/reporting the device

import dataset                                                                 #Config loading, session/window reporting
import train                                                                   #The training loop itself


def main():
    run_start = time.perf_counter()                                           #Marking when the run started
    print("=" * 70)                                                           #Opening banner rule
    print("[main] Fingertip force estimator - training run")                  #Announcing what this run is
    print("=" * 70)                                                           #Closing banner rule

    config = dataset.load_config()                                            #Loading config.yaml
    model_config = config["model_training"]                                   #Pulling out the model_training block
    splits = model_config["splits"]                                           #Pulling out the split assignment
    loader_config = model_config["loader"]                                    #Pulling out the loader settings
    training_config = model_config["training"]                                #Pulling out the training settings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     #Picking the device
    device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"  #Naming it for the log
    print(f"[main] Device: {device} ({device_label})")                        #Reporting the device in use
    print(f"[main] Train sessions ({len(splits['train_sessions'])}): {splits['train_sessions']}")  #Reporting the train split
    print(f"[main] Val session:  {splits['val_sessions']}")                   #Reporting the val split
    print(f"[main] Test session: {splits['test_sessions']}")                  #Reporting the held-out test split
    print(f"[main] Batch size: {loader_config['batch_size']}  "
          f"Effective batch size: {training_config['effective_batch_size']}")  #Reporting batch settings
    print(f"[main] Max epochs: {training_config['max_epochs']}  "
          f"Early-stopping patience: {training_config['early_stopping_patience']}")  #Reporting the training budget

    print("[main] Handing off to train.train() for dataset loading, model build, and the training loop")  #Announcing the handoff
    best_val_mae_n = train.train(config)                                      #Running dataset build + model build + full training loop

    elapsed_min = (time.perf_counter() - run_start) / 60.0                    #Computing total run time in minutes
    checkpoint_dir = training_config["checkpoint_dir"]                        #Looking up where the checkpoint was written
    metrics_file = training_config["metrics_history_file"]                    #Looking up the metrics history filename

    print("=" * 70)                                                          #Opening summary rule
    print(f"[main] Training run finished in {elapsed_min:.1f} minutes")       #Reporting total run time
    print(f"[main] Best validation MAE: {best_val_mae_n:.3f} N")              #Reporting the headline result
    print(f"[main] Checkpoint written to: {checkpoint_dir}/best_model.pt")    #Reporting the checkpoint location
    print(f"[main] Metrics history written to: {checkpoint_dir}/{metrics_file}")  #Reporting the metrics history location
    print("[main] Open explore.ipynb to inspect training curves and run the test-set evaluation")  #Pointing to the next step
    print("=" * 70)                                                          #Closing summary rule


if __name__ == "__main__":                                                    #Standard entry-point guard
    main()                                                                    #Running the pipeline
