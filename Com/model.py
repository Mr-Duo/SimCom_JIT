import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaModel, BertModel


class CodeBERT_JIT(nn.Module):
    def __init__(self, args):
        super(CodeBERT_JIT, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        Dim = args.embed_size
        Class = args.class_num
        Num_feature = args.num_feature        

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # For commit message
        self.BERT = BertModel.from_pretrained("bert-base-uncased")

        # For commit code change
        self.codeBERT = RobertaModel.from_pretrained("microsoft/codebert-base")

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(768 * 3 + Num_feature, args.hidden_size)  # hidden units
        self.fc2 = nn.Linear(args.hidden_size, Class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, added_code, removed_code, feature, message):
        x_added_coded = self.codeBERT(added_code)
        x_removed_coded = self.codeBERT(removed_code)
        x_message = self.BERT(message)

        x_added_coded = x_added_coded[0][:, 0]
        x_removed_coded = x_removed_coded[0][:, 0]
        x_message = x_message[0][:, 0]

        x_commit = torch.cat((x_added_coded, x_removed_coded, x_message), 1)
        x_commit = self.dropout(x_commit)
        x_commit = torch.cat((x_commit, feature), 1)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out