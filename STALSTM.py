import numpy as np
import torch
import visdom
from torch.nn import LSTM, STALSTM, Sequential, Linear, Dropout, Tanh
import torch.optim as optim
import os
from read_data import NTUDataSet

class LSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMNet, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, 3)
        self.fc = Linear(hidden_size, 60)
        self.hidden_size = hidden_size
        self.input_size = input_size

    def forward(self, input, hx=None):
        h0 = torch.randn(3, input.size(1), self.hidden_size)
        c0 = torch.randn(3, input.size(1), self.hidden_size)
        output, (hn, cn) = self.lstm(input, (h0, c0))
        result = torch.tanh(self.fc(output.sum(0)))

        return result


class STALSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(STALSTMNet, self).__init__()

        self.mainlstm = LSTM(input_size, hidden_size, 3, bidirectional=True)
        self.salstm = STALSTM(input_size, hidden_size)
        self.mainfc = Sequential(Linear(hidden_size*2, 60))
        #body part for NTU RGB+D
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.body_part_size = 8
        self.body_part_index = {}
        self.body_part_value = {}

    def forward(self, input, hx=None):
        """if hx is None:
            h_0 = input.data.new(2, input.size(1), self.hidden_size).fill_(0)
            c_0 = input.data.new(2, input.size(1), self.hidden_size).fill_(0)
            hx_sa = (h_0.clone(), c_0.clone())
            h_0 = input.data.new(6, input.size(1), self.hidden_size).fill_(0)
            c_0 = input.data.new(6, input.size(1), self.hidden_size).fill_(0)
            hx_main = (h_0.clone(), c_0.clone())
        else:
            hx_sa = hx.clone()
            hx_main = hx.clone()
        sa_output, sa_hn = self.salstm(input, hx_sa)
        sa_hn = torch.cat((sa_hn[0], sa_hn[1]), 1)
        for i in range(sa_hn.size(0)):
            normalize_input = torch.exp(self.safc(sa_hn[i, :]))

            selected_value, selected = normalize_input.topk(5)
            selected_value.fill_(selected_value.mean())
            selected = set(torch.Tensor.tolist(selected))

            selected_value = dict(zip(list(selected), list(selected_value)))
            if self.body_part_index == {}:
                self.body_part_index = selected
                self.body_part_value = selected_value
            else:
                if selected.intersection(self.body_part_index) != set():
                    inter_id = selected.intersection(self.body_part_index)
                    for t, sel_item in enumerate(selected):
                        if sel_item in inter_id:
                            self.body_part_value[sel_item] = self.body_part_value[sel_item] + \
                                                             selected_value[sel_item]
                        else:
                            self.body_part_value[sel_item] = selected_value[sel_item]
                    self.body_part_index = self.body_part_index.union(selected)
                else:
                    self.body_part_index = self.body_part_index.union(selected)
                    self.body_part_value.update(selected_value)
                    break

        t_sum = np.array(list(self.body_part_value.values())).sum()
        for key, value in self.body_part_value.items():
            self.body_part_value[key] = self.body_part_value[key] / t_sum

        alpha = torch.ones(75) * 0.001
        for key, value in self.body_part_value.items():
            alpha[key*3] = self.body_part_value[key]
            alpha[key*3+1] = self.body_part_value[key]
            alpha[key*3+2] = self.body_part_value[key]

        alpha.view(1, 1, -1)
        input = input * alpha
        main_output, (hn, cn) = self.mainlstm(input, hx_main)
        hn = torch.cat((hn[4], hn[5]), 1)
        output = self.mainfc(hn)

        return output"""
        h0 = torch.randn(1, input.size(1), self.hidden_size)
        c0 = torch.randn(1, input.size(1), self.hidden_size)
        input_w, (hn, cn) = self.salstm(input, (h0, c0))
        input = input[1:].mul(input_w)
        h0 = torch.randn(6, input.size(1), self.hidden_size)
        c0 = torch.randn(6, input.size(1), self.hidden_size)
        output, (hn, cn) = self.mainlstm(input, (h0, c0))
        result = torch.sum(output, 0)
        result = self.mainfc(result)
        #result = torch.tanh(result)

        return result


class STALoss(torch.nn.Module):
    def __init(self, lambda1, lambda2):
        super(STALoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, output, truth):
        crossen = torch.nn.CrossEntropyLoss()
        corssen_loss = crossen(output, truth)


if __name__ == '__main__':
    vis = visdom.Visdom(env=u'train')
    model = STALSTMNet(3*25, 128)
    for epoch in range(1000):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        train_data = NTUDataSet(train=True, cross_subject=True, cross_view=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=256,
                                                   shuffle=True, num_workers=4)
        train_correct = 0.0
        train_acc = []
        for batch_idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            input_data = data.permute([1, 0, 2])
            score = model(input_data)
            _,           index = torch.max(score, 1)
            label = label - 1
            train_correct += (index == label).sum()
            train_acc.append(int(train_correct) / 256)
            loss = criterion(score, label)
            print('epoch :%d batch :%d loss :%f correct :%d' % (epoch, batch_idx, loss, (index==label).sum()))
            loss.backward(retain_graph=True)
            optimizer.step()
        train_acc = torch.Tensor(train_acc).mean()
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([train_acc]), win='acc', opts=dict(title='acc'),
                 update='append' if epoch > 0 else None)
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([loss]), win='loss', opts=dict(title='loss'),
                 update='append' if epoch > 0 else None)
        torch.save(model.state_dict(), 'param.pkl')




