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
        self.body_part_value = [dict()]

    def forward(self, input, hx):
        x = hx.copy()
        sa_output, (sa_hn, sa_cn) = self.salstm(input, hx)
        alpha = []
        for i in range(sa_hn.size(1)):
            normalize_input = torch.exp(self.safc2(torch.tanh(self.safc1(sa_hn[:,i,:]))))
            normalize_input /= normalize_input.sum()
            selected = set([i for i, x in enumerate(normalize_input) if x >= 0.1])
            selected_value = dict([zip(x, normalize_input[x]) for x in selected])
            for j, subset in enumerate(self.body_part_index):
                if selected & subset is not []:
                    inter_id = selected.intersection(subset)
                    #set_id = [list(self.body_part_index[j]).index(id) for id in inter_id]
                    #selected_id = [list(selected).index(id) for id in inter_id]
                    #inter_value = [selected_value[i] for i in selected_id]
                    for t, sel_item in enumerate(selected):
                        if sel_item in inter_id:
                            self.body_part_value[j][sel_item] += selected_value[sel_item]
                        else:
                            self.body_part_value[j][sel_item] = selected_value[sel_item]
                elif j == self.body_part_index.len() - 1:
                    self.body_part_index.append(selected)
                    self.body_part_value.append(dict())
                    for key, value in enumerate(selected):
                        self.body_part_value[-1][value] = selected_value[value]

                t_sum = 0
                for t, in enumerate(self.body_part_value):
                    t_sum += np.array(list(self.body_part_value[t].values())).sum()
                for t, in enumerate(self.body_part_value):
                    for key, value in self.body_part_value[t].items():
                        self.body_part_value[t][key] /= t_sum


            alpha.append(noramlize_input)

        input *= alpha
        main_output, (hn, cn) = self.mainlstm(input, x)
        output = self.mainfc1(hn)

        return output
