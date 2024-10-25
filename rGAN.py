import torch
import torch.nn as nn
from unet_parts import *
from torch.autograd import Variable
from typing import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size: (int, int), input_dim: int, hidden_dim:int, kernel_size:(int, int), bias:bool) -> None:
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor: torch.Tensor, cur_state: [torch.Tensor, torch.Tensor]) -> [torch.Tensor, torch.Tensor]:
        # print(cur_state, ' -------')
        h_cur, c_cur = cur_state

        # print('ConvLSTMCell forward --- type of h_cur: ', type(h_cur), h_cur.shape, ' type of c_cur: ', type(c_cur), c_cur.shape)
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # print('ConvLSTMCell forward --- type of combined: ', type(combined))
        
        combined_conv = self.conv(combined)

        # print('ConvLSTMCell forward --- type of combined_cov: ', type(combined_conv))


        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # print('ConvLSTMCell forward --- cc_i: ', type(cc_i), cc_i.shape, ' cc_f: ', type(cc_f), cc_f.shape, ' cc_o: ', type(cc_o), cc_o.shape, ' cc_g: ', type(cc_g), cc_g.shape)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # print('ConvLSTMCell forward --- i: ', type(i), i.shape, ' f: ', type(f), f.shape, ' o: ', type(o), o.shape,
        #       ' g: ', type(g), g.shape)

        c_next = f * c_cur + i * g

        # print('ConvLSTMCell forward --- c_next: ', type(c_next), c_next.shape)
        h_next = o * torch.tanh(c_next)
        # print('ConvLSTMCell forward --- h_next: ', type(h_next), h_next.shape)
        return h_next, c_next

    def init_hidden(self, batch_size: int) -> (torch.Tensor, torch.Tensor):

        # return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
        #         Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).to(device),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size:int, input_dim:int, hidden_dim:int, kernel_size:int, num_layers:int,
                 batch_first=False, bias=True) -> None:
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            ks = self.kernel_size[i]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=ks,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor: torch.Tensor, hidden_state:list[torch.Tensor]) -> (list[torch.Tensor], torch.Tensor):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            4-D Tensor either of shape (t, c, h, w) 
            
        Returns
        -------
        last_state_list, layer_output
        """
        
        cur_layer_input = input_tensor

        # print(hidden_state, ' ---------- lei -------')
        
        for layer_idx in range(self.num_layers):
            output_inner=[]
            h,c = hidden_state[layer_idx]
            # print('ConvLSTM forward --- h: ', type(h), h.shape, ' c: ', type(c), c.shape)
            h,c = self.cell_list[layer_idx](cur_layer_input, cur_state=[h, c])
            # print('ConvLSTM forward --- h: ', type(h), h.shape, ' c: ', type(c), c.shape)
            hidden_state[layer_idx] = h,c
            cur_layer_input = h
            # print('ConvLSTM forward --- cur_layer_input: ', type(cur_layer_input), cur_layer_input.shape)
        return h, hidden_state
    
    def _init_hidden(self, batch_size: int) -> list[torch.Tensor]:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        # print(init_states, ' ++++++ ')
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

def double_conv(in_channels: int, out_channels:int) -> None:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )  

class Generator(nn.Module):
    def __init__(self, batch_size: int) -> None:
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.have_cuda = True
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 3)
        self.ConvLSTM = ConvLSTM(input_size=(256,256),
                 input_dim= 3,
                 hidden_dim=[16,128, 256,128, 64, 32,16,3],
                 kernel_size=(3, 3),
                 num_layers=8, 
                 batch_first=True,
                 bias=True)
        
    def forward(self,x: list[torch.Tensor]) -> torch.Tensor:
        for t in range(3):
            hidden_state = self.ConvLSTM._init_hidden(batch_size=self.batch_size)
            x1 = self.inc(x[t])
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            # print('Generator forward--- x1: ', type(x1), x1.shape, 'x2: ', type(x2), x2.shape, 'x3: ', type(x3), x3.shape, 'x4: ', type(x4), x4.shape, 'x5: ', type(x5), x5.shape)

            recon_x = self.up1(x5, x4)
            recon_x = self.up2(recon_x, x3)
            recon_x = self.up3(recon_x, x2)
            recon_x = self.up4(recon_x, x1)
            recon_x = self.outc(recon_x)
            # print('Generator forward--- recon_x: ', type(recon_x), recon_x.shape)

            h,hidden_state=self.ConvLSTM(recon_x, hidden_state)
            # print('Generator forward--- h: ', type(h), h.shape, 'hidden_state: ', type(hidden_state), len(hidden_state))
        return h


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.have_cuda = True
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(True),
        )
        self.adv_layer =  nn.Sequential( nn.Linear(128*16*16, 1),
                                        nn.Sigmoid())
    

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.discriminator(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        # print('Discriminator forward--- out: ', type(out), out.shape, 'validity: ', type(validity), validity.shape)
  
        return validity

class GDL(nn.Module):   
    def __init__(self, pNorm=2):
        
        super(GDL, self).__init__()
        self.convX = nn.Conv2d(3, 3, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(3, 3, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)
        filterX = torch.Tensor(torch.FloatTensor([[[[-1, 1]], [[-1, 1]], [[-1, 1]]]]))
        filterY = torch.Tensor(torch.FloatTensor([[[[1], [-1]],[[1], [-1]], [[1], [-1]]]]))

        self.convX.weight = torch.nn.Parameter(filterX.cuda(), requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY.cuda(), requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt):
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())
        
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))
        
        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        
        
        mat_loss_x = grad_diff_x ** self.pNorm
        
        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height
        
        shape = gt.shape

        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (shape[0] * shape[1] * shape[2] * shape[3]) 
               
        return mean_loss