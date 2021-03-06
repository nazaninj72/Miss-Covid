import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from fcclassifier import *
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from functions import *
import argparse
import torch
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('output.log')
fh.setLevel(logging.DEBUG) # or any level you want
logger.addHandler(fh)


parser = argparse.ArgumentParser()
## learning
parser.add_argument('--epoch', type=int, default=20, help='number of epochs for train [default: 20]')
parser.add_argument('--random_seed', type=int, default=42, help='random_seed [default: 42]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 32]')
# parser.add_argument('-hidden-dim', type=int, default=768, help='number of hidden dimension [default: 768]')
## view
parser.add_argument('--with_metadata',action="store_true", help='Whether to use metadata features')
parser.add_argument('--learningrate', type=float, default=5e-5, help='learning rate [default: 5e-5]')
# parser.add_argument('-','-max_len', type=int, default=160, help='maximum length of sentences [default: 160]')
# parser.add_argument('-reply_sample_frequency', type=int, default=1, help='1:without sample')
parser.add_argument(
        '--model',
        type=str,
        default='bert',
        choices=['bert', 'ct-bert'],
        help='Type of models [default: bert]'
    )
parser.add_argument(
        '--dataset',
        type=str,
        default='flu-claims',
        choices=['flu-claims', 'topics'],
        help='Dataset to classify misinformation [default: flu-claims]'
    )
parser.add_argument(
        '--topic',
        type=str,
        choices=['politics', 'other','immunity','transmission','health'],
        help='Topics of misinformation  [choices: politics, other,immunity,transmission,health]'
    )
args = parser.parse_args()





if torch.cuda.is_available():       
    device = torch.device("cuda")
    # print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    # print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# print(args.max_length)



# Specify loss function
loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4,with_metadata=False,evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} |{'Train Acc':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            # Zero out any previously calculated gradients
            model.zero_grad()
            if with_metadata: 
                b_input_ids, b_attn_mask,b_md, b_labels = tuple(t.to(device) for t in batch)
                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask,b_md)
                

            else: 
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)



                

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy,_ = evaluate(model, val_dataloader,with_metadata)
            _,train_accuracy,_=evaluate(model,train_dataloader,with_metadata)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {train_accuracy:^9.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader,with_metadata=False):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    preds=[]
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        if with_metadata: 
            b_input_ids, b_attn_mask,b_md, b_labels = tuple(t.to(device) for t in batch)
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask,b_md)
            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask,b_md)
            

        else:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)


            


        

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        pred= torch.argmax(logits, dim=1).flatten()
        preds.append(pred)
        # Calculate the accuracy rate
        accuracy = (pred == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy,preds







if __name__ == "__main__":
    MAX_LENGTH=160

    epochs=args.epoch
    batch_size=args.batch_size
    model_type=args.model
    learningrate=args.learningrate
    with_metadata = args.with_metadata
    dataset = args.dataset
    topic = args.topic
    print(topic)
    if dataset=='flu-claims':
        path_to_data='/home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/data/'+dataset+'.tsv'
    else:
        path_to_data='/home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/data/'+topic+'.tsv'
    data = pd.read_csv(path_to_data)
    print("Experiment details: dataset "+dataset+   " #epochs "+str(epochs)+" batch size "+str(batch_size)+" learning rate  "+str(learningrate)+" model "+model_type+" with metadata "+str(with_metadata))
    X = data.tweet.values
    y = data.label.values
    random_state=args.random_seed

    indices = np.arange(len(X))
    train_idx, val_idx, y_train, y_val= train_test_split(indices, y,stratify = y, test_size=0.1, random_state=random_state)
    X_train = X[train_idx]
    X_val = X[val_idx]
    

    train_inputs, train_masks = preprocessing_for_bert(X_train,MAX_LENGTH,model_type)
    val_inputs, val_masks = preprocessing_for_bert(X_val,MAX_LENGTH,model_type)

    

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # Create the DataLoader for our training set
    if with_metadata:
        # # Create the DataLoader for our training set
        # if dataset=='topics':
        #     path_to_metadata='/home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/data/'+topic+'-metadata.tsv'
        # else:
        #     path_to_metadata='/home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/data/'+dataset+'-metadata.tsv'
        # meta_data=pd.read_csv(path_to_metadata)
        
        # meta_data=get_metadata(data)
        md_X=get_metadata_features(data)
        mdX_train = md_X[train_idx]
        mdX_val = md_X[val_idx]
        md_train=torch.tensor(mdX_train).type(torch.FloatTensor)
        md_val=torch.tensor(mdX_val).type(torch.FloatTensor)
        train_data = TensorDataset(train_inputs, train_masks,md_train, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks,md_val, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

        
    else:
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        

    set_seed(42)    # Set seed for reproducibility
    ctbert_classifier, optimizer, scheduler = initialize_model(train_dataloader,epochs,learningrate,with_metadata,model_type)
    train(ctbert_classifier, train_dataloader, val_dataloader, epochs,with_metadata,evaluation=True)
    val_loss, val_accuracy,y_pred= evaluate(ctbert_classifier, val_dataloader,with_metadata)
    np_preds=[]
    for i in y_pred:
        b=i.cpu().detach().numpy()
        np_preds=np.append(np_preds,b,axis=0)
    np_preds=np_preds.astype(int) 
    class_names = ['fake', 'real']
    print(classification_report(y_val, np_preds, target_names=class_names))

