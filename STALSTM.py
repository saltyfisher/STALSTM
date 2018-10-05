import numpy as np
import torch
from torch.nn import LSTM, Sequential, Linear
from torch.autograd.variable import Variable

class STALSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(STALSTM, self).__init__()

        self.mainlstm = LSTM(input_size, hidden_size, 3, bidirectional=True)
        self.salstm = LSTM(input_size, hidden_size, bidirectional=True)
        self.safc1 =  Linear(hidden_size, hidden_size)
        self.safc2 = Linear(hidden_size, input_size[0].size(1))
        self.mainfc1 = Linear(hidden_size, hidden_size)
        #body part for NTU RGB+D
        self.body_part_size = 8
        self.body_part_index = [{1, 2, 21, 3, 4}, {9, 10, 11, 12, 25, 24}, {5, 6, 7, 8, 23, 22},
                                {17, 18, 19, 20}, {13, 14, 15, 16}]
        self.body_part_value = []

    def forward(self, input, hx):
        x = hx.copy()
        sa_output, (sa_hn, sa_cn) = self.salstm(input, hx)
        alpha = []
        for i in range(sa_hn.size(1)):
            noramlize_input = torch.exp(self.safc2(torch.tanh(self.safc1(sa_hn[:,i,:]))))
            noramlize_input /= noramlize_input.sum()
            selected = set([i for i, x in noramlize_input if x >= 0.1])
            selected_value = [noramlize_input[x] for x in selected]
            if selected not in self.body_part_index and self.body_part_index.len() < self.body_part_size:
                self.body_part_index.append(selected)
            else:
                for j in self.body_part_index.len():
                    if selected & self.body_part_index is not []:
                        inter_id = selected.intersection(self.body_part_index[j])
                        set_id = [list(self.body_part_index[j]).index(id) for id in inter_id]
                        for t, id in enumerate(set_id):
                            self.body_part_value[j][id] += selected_value[t]
                            selected_value[t] = self.body_part_value[j][id]

                        diff_id = selected.difference(self.body_part_index[j])
                        if diff_id is not []:
                            new_mean = selected_value.sum()

            selected_mean = torch.mean([noramlize_input[i] for i in selected])
            for i in selected:
                noramlize_input[i] = selected_mean
            alpha.append(noramlize_input)

        input *= alpha
        main_output, (hn, cn) = self.mainlstm(input, x)
        output = self.mainfc1(main_output)

        return output
