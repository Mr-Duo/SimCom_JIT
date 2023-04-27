import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaModel


class CodeBERT_JIT(nn.Module):
    def __init__(self, args):
        super(CodeBERT_JIT, self).__init__()
        self.args = args

        Class = args.class_num

        self.codeBERT = RobertaModel.from_pretrained("microsoft/codebert-base")
        for param in self.codeBERT.base_model.parameters():
            param.requires_grad = False

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(768 * 2, args.hidden_size)  # hidden units
        self.fc2 = nn.Linear(args.hidden_size, Class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, added_code, removed_code):
        x_added_coded = self.codeBERT(added_code)
        x_removed_coded = self.codeBERT(removed_code)

        x_added_coded = x_added_coded[0][:, 0]
        x_removed_coded = x_removed_coded[0][:, 0]

        x_commit = torch.cat((x_added_coded, x_removed_coded), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out