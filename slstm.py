from mindspore import nn, Tensor
from mindspore import Parameter
import mindspore.ops as ops
import numpy as np

class LayerNorm(nn.Cell):
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = Parameter(Tensor(np.ones(features)), requires_grad=True)
        self.b_2 = Parameter(Tensor(np.zeros(features)), requires_grad=True)
        self.eps = eps

    def construct(self, x):
        mean = ops.ReduceMean()(x, -1, keep_dims=True)
        std = ops.Sqrt()(ops.ReduceMean()(ops.Square()(x - mean), -1, keep_dims=True))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
      
class SLSTM(nn.Cell):
    def __init__(self, config):
        super(SLSTM, self).__init__()
        self.config = config
        hidden_size = config.HP_hidden_dim
        print(f"hidden_size: {hidden_size}")
        #sys.exit(0)

        # forget gate for left
        self.Wxf1, self.Whf1, self.Wif1, self.Wdf1 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #forget gate for right
        self.Wxf2, self.Whf2, self.Wif2, self.Wdf2 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #forget gate for inital states
        self.Wxf3, self.Whf3, self.Wif3, self.Wdf3 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #forget gate for dummy states
        self.Wxf4, self.Whf4, self.Wif4, self.Wdf4 = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #input gate for current state
        self.Wxi, self.Whi, self.Wii, self.Wdi = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)
        #input gate for output gate
        self.Wxo, self.Who, self.Wio, self.Wdo = self.create_a_lstm_gate(hidden_size, self.config.HP_gpu)

        self.bi = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bo = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf1 = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf2 = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf3 = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.bf4 = self.create_bias_variable(hidden_size, self.config.HP_gpu)

        self.gated_Wxd = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)
        self.gated_Whd = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)

        self.gated_Wxo = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)
        self.gated_Who = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)

        self.gated_Wxf = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)
        self.gated_Whf = self.create_to_hidden_variable(hidden_size, hidden_size, self.config.HP_gpu)

        self.gated_bd = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.gated_bo = self.create_bias_variable(hidden_size, self.config.HP_gpu)
        self.gated_bf = self.create_bias_variable(hidden_size, self.config.HP_gpu)

        self.h_drop = nn.Dropout(config.HP_dropout)
        self.c_drop = nn.Dropout(config.HP_dropout)

        self.g_drop = nn.Dropout(config.HP_dropout)

        #self.input_norm = LayerNorm(hidden_size)
        self.i_norm = LayerNorm(hidden_size)
        self.o_norm = LayerNorm(hidden_size)
        self.f1_norm = LayerNorm(hidden_size)
        self.f2_norm = LayerNorm(hidden_size)
        self.f3_norm = LayerNorm(hidden_size)
        self.f4_norm = LayerNorm(hidden_size)

        self.gd_norm = LayerNorm(hidden_size)
        self.go_norm = LayerNorm(hidden_size)
        self.gf_norm = LayerNorm(hidden_size)


    def create_bias_variable(self, size, gpu=False, mean=0.0, stddev=0.1):
        data = np.zeros(size, dtype=np.float32)
        var = Parameter(Tensor(data), requires_grad=True)
        var.set_data(Tensor(np.random.normal(mean, stddev, size).astype(np.float32)))
        return var

    def create_to_hidden_variable(self, size1, size2, gpu=False, mean=0.0, stddev=0.1):
        data = np.zeros((size1, size2), dtype=np.float32)
        var = Parameter(Tensor(data), requires_grad=True)
        var.set_data(Tensor(np.random.normal(mean, stddev, (size1, size2)).astype(np.float32)))
        return var
