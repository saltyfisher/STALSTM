import numpy as np
import torch
from torch.nn import LSTM, Sequential, Linear
import torch.optim as optim

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
        self.body_part_index = [{}]
        self.body_part_value = [dict()]

    def forward(self, input, hx):
        x = hx.copy()
        sa_output, (sa_hn, sa_cn) = self.salstm(input, hx)
        alpha = []
        for i in range(sa_hn.size(1)):
            normalize_input = torch.exp(self.safc2(torch.tanh(self.safc1(sa_hn[:,i,:]))))
            normalize_input /= normalize_input.sum()
            selected = {}
            for j, x in enumerate(normalize_input):
                if x >= 0.1 & selected.len() <= 5:
                    selected.add(j)

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
                for t, p in enumerate(self.body_part_value):
                    t_sum += np.array(list(self.body_part_value[t].values())).sum()
                for t, in enumerate(self.body_part_value):
                    for key, value in self.body_part_value[t].items():
                        self.body_part_value[t][key] /= t_sum

            t_alpha = np.ones(25) * 0.001
            for t, p in enumerate(self.body_part_value):
                for key, value in self.body_part_value[t].items():
                    t_alpha[key] = self.body_part_value[t][key]

            alpha.append(t_alpha)

        input *= alpha
        main_output, (hn, cn) = self.mainlstm(input, x)
        output = self.mainfc1(hn)

        return output

class STALoss(torch.nn.Module):
    def __init(self, lambda1, lambda2):
        super(STALoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, output, truth):
        crossen = torch.nn.CrossEntropyLoss()
        corssen_loss = crossen(output, truth)


if __name__ == '__main__':
    pass



