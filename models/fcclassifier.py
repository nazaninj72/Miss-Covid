# %%time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
# model = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

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
    def __init__(self):
        super(MergedClassifier, self).__init__()
        self.BertClassifier = BertClassifier(freeze_bert=False)
        self.FeatureClassifier = FeatureClassifier(4)
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, input_ids, attention_mask, x2):
        x1 = self.BertClassifier(input_ids, attention_mask)
        x2 = self.FeatureClassifier(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x