# %%time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_curve, auc
from functions import *
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
learningrate=5e-5
# Create a function to tokenize a set of texts
import torch.nn.functional as F



if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")






def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs



def preprocessing_for_bert(data,MAX_LEN=160):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,     # Return attention mask
            truncation=True
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks




def initialize_model(train_dataloader,epochs=4,lr=learningrate,classifier_type='bert',model='bert'):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    if classifier_type=='bert':
        # Instantiate Bert Classifier
        classifier = BertClassifier(model,freeze_bert=False)
        # Tell PyTorch to run the model on GPU
        classifier.to(device)
    elif classifier_type == 'bert_metadata':
        classifier=MergedClassifier(model,classifier_type)
        classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(classifier.parameters(),
                      lr=learningrate,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return classifier, optimizer, scheduler

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, model='bert', freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
         if model=='bert':
            D_in, H, D_out = 768, 50, 2
            # Instantiate BERT model
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            D_in, H, D_out = 1024,8, 2
            # Instantiate BERT model
            self.bert = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
#         print("input_ids",input_ids.size())
#         print("attention_mask",attention_mask.size())
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    
    
    
class FeatureClassifier(nn.Module):
    def __init__(self,D_in=4):
        super(FeatureClassifier, self).__init__()
        self.D_in=D_in
        self.fc1 = nn.Linear(self.D_in, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        return x


class MergedClassifier(nn.Module):
    def __init__(self,f_dim=4,model='bert'):
        super(MergedClassifier, self).__init__()
        self.BertClassifier = BertClassifier(model='bert',freeze_bert=False)
        self.FeatureClassifier = FeatureClassifier(f_dim)
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, input_ids, attention_mask, x2):
        x1 = self.BertClassifier(input_ids, attention_mask)
        x2 = self.FeatureClassifier(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x