import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils.ml_utils import ModelLogger, NoopModelLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.model_type = 'Transformer'
        self.input_ff = nn.Linear(input_features, d_model)
        self.input_ff_sigmoid = nn.Sigmoid()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.final_linear = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_ff.bias.data.zero_()
        self.input_ff.weight.data.uniform_(-initrange, initrange)
        self.final_linear.bias.data.zero_()
        self.final_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None, logger: ModelLogger = NoopModelLogger()) -> Tensor:
        # src: [batch_size b, seq_len k 55, features 27]
        output = src
        logger.log(f"{output.size()}")
        output = self.input_ff(output)
        output = self.input_ff_sigmoid(output)
        logger.log(f"input_ff - {output.size()}")
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(dim=1)).to(device)
        output = self.transformer_encoder(output, src_mask)
        logger.log(f"encoder - {output.size()}")
        output = self.final_linear(output)
        logger.log(f"final_linear - {output.size()}")
        output = torch.squeeze(output, dim=-1)
        logger.log(f"output - {output.size()}")
        return output
