import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaModel


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

        self.codeBERT = RobertaModel.from_pretrained("microsoft/codebert-base")

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(768 * 2 + len(Ks) * Co, args.hidden_size)  # hidden units
        self.fc2 = nn.Linear(args.hidden_size + Num_feature, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward(self, added_code, removed_code, feature, message):
        x_added_coded = self.codeBERT(added_code)
        x_removed_coded = self.codeBERT(removed_code)
        x_message = self.embed_msg(message)
        x_message = self.forward_msg(x_message, self.convs_msg)

        x_added_coded = x_added_coded[0][:, 0]
        x_removed_coded = x_removed_coded[0][:, 0]

        x_commit = torch.cat((x_added_coded, x_removed_coded, x_message), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = torch.cat((out, feature), 1)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out