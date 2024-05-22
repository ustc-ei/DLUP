<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F


class identification_model(nn.Module):
    def __init__(self, lib_mz_num, ms_size):
        super(identification_model, self).__init__()
        self.lib_mz_num = lib_mz_num
        self.ms_size = ms_size

        self.conv1_lib = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.conv2_lib = nn.Conv1d(64, 128, kernel_size=1, padding=0)
        self.conv3_lib = nn.Conv1d(128, 256, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.conv1_ms = nn.Conv1d(ms_size, 64, kernel_size=1, padding=0)
        self.conv2_ms = nn.Conv1d(64, 128, kernel_size=1, padding=0)
        self.conv3_ms = nn.Conv1d(128, 256, kernel_size=1, padding=0)
        self.conv1_merge = nn.Conv1d(256*4, 256, kernel_size=1, padding=0)
        a = lib_mz_num
        self.fc1_merge = nn.Linear(256 * a, 256)

        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, matchedMs2, Spectrum,  Ms2IonMobility, IonMobility):
        matchedMs2 = matchedMs2[:, :, :, 1]
        Spectrum = Spectrum[:, :, :, 1]
        Spectrum = F.relu(self.conv1_lib(Spectrum))
        Spectrum = F.relu(self.conv2_lib(Spectrum))
        Spectrum = F.relu(self.conv3_lib(Spectrum))
        Spectrum = self.dropout(Spectrum)

        matchedMs2 = F.relu(self.conv1_ms(matchedMs2))
        matchedMs2 = F.relu(self.conv2_ms(matchedMs2))
        matchedMs2 = F.relu(self.conv3_ms(matchedMs2))
        matchedMs2 = self.dropout(matchedMs2)

        Ms2IonMobility = F.relu(self.conv1_ms(Ms2IonMobility))
        Ms2IonMobility = F.relu(self.conv2_ms(Ms2IonMobility))
        Ms2IonMobility = F.relu(self.conv3_ms(Ms2IonMobility))
        Ms2IonMobility = self.dropout(Ms2IonMobility)

        IonMobility = F.relu(self.conv1_lib(IonMobility))
        IonMobility = F.relu(self.conv2_lib(IonMobility))
        IonMobility = F.relu(self.conv3_lib(IonMobility))
        IonMobility = self.dropout(IonMobility)

        z = torch.cat((Spectrum, matchedMs2, IonMobility, Ms2IonMobility), 1)
        z = F.relu(self.conv1_merge(z))
        z = self.dropout(z)
        z = z.view(z.shape[0], -1)
        z = F.relu(self.fc1_merge(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        z = self.dropout(z)
        z = torch.sigmoid(self.fc3(z))
        return z
    
class quantification_model(nn.Module):
    def __init__(self, lib_mz_num, ms_size):
        super(quantification_model, self).__init__()
        self.lib_mz_num = lib_mz_num
        self.ms_size = ms_size

        self.conv1_lib = nn.Conv1d(1, 64, kernel_size=1, padding=0)
        self.conv2_lib = nn.Conv1d(64, 128, kernel_size=1, padding=0)
        self.conv3_lib = nn.Conv1d(128, 256, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(0.1)
        self.conv1_ms = nn.Conv1d(ms_size, 64, kernel_size=1, padding=0)
        self.conv2_ms = nn.Conv1d(64, 128, kernel_size=1, padding=0)
        self.conv3_ms = nn.Conv1d(128, 256, kernel_size=1, padding=0)
        self.conv1_merge = nn.Conv1d(256*2, 256, kernel_size=1, padding=0)
        a = lib_mz_num
        self.fc1_merge = nn.Linear(256 * a, 256)

        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, Spectrum, matchedMs2):
        matchedMs2 = matchedMs2[:, :, :, 1]
        Spectrum = Spectrum[:, :, :, 1]
        Spectrum = F.relu(self.conv1_lib(Spectrum))
        Spectrum = F.relu(self.conv2_lib(Spectrum))
        Spectrum = F.relu(self.conv3_lib(Spectrum))
        Spectrum = self.dropout(Spectrum)

        matchedMs2 = F.relu(self.conv1_ms(matchedMs2))
        matchedMs2 = F.relu(self.conv2_ms(matchedMs2))
        matchedMs2 = F.relu(self.conv3_ms(matchedMs2))
        matchedMs2 = self.dropout(matchedMs2)

        z = torch.cat((Spectrum, matchedMs2), 1)
        z = F.relu(self.conv1_merge(z))
        z = self.dropout(z)
        z = z.view(z.shape[0], -1)
        z = F.relu(self.fc1_merge(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        z = self.dropout(z)
        z = F.relu(self.fc3(z))

        return z
=======
from enum import Enum, auto

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer
import copy


class CALCULATE_METHOD(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    MERGE = auto()


class identify_MLP(nn.Module):
    def __init__(self, d_out):
        super(identify_MLP, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.SELU(),
            nn.Linear(d_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class QuantMLP(nn.Module):
    def __init__(self, d_out):
        super(QuantMLP, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.SELU(),
            nn.Linear(d_out, 1),
            # nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class IdentifyModel(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout=0.1,
        num_encoder_layers: int = 6,
        calculate_method=CALCULATE_METHOD.SUB
    ) -> None:
        super(IdentifyModel, self).__init__()
        self.calculate_method = calculate_method

        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_out,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu"
        )

        self.ms_trans_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )

        self.pep_trans_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )

        self.ms_encoder = nn.Sequential(
            nn.LayerNorm(6, ),
            nn.Linear(d_in, d_out),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        self.pep_encoder = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        self.general_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )

        self.encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )

        self.decoder = identify_MLP(d_out)

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def merge(x, y):
        return torch.concat((x, y), dim=1)

    @staticmethod
    def mul(x, y):
        return torch.mul(x, y)

    @staticmethod
    def sub(x, y):
        return x - y

    def consolidate(self, matchedMs2, Spectrum):
        if self.calculate_method == CALCULATE_METHOD.ADD:
            x = self.add(matchedMs2, Spectrum)
        elif self.calculate_method == CALCULATE_METHOD.SUB:
            x = self.sub(matchedMs2, Spectrum)
        elif self.calculate_method == CALCULATE_METHOD.MUL:
            x = self.mul(matchedMs2, Spectrum)
        elif self.calculate_method == CALCULATE_METHOD.MERGE:
            x = self.merge(matchedMs2, Spectrum)

        return x

    def forward(self, matchedMs2, Spectrum):
        # batch * matchedNum * peaksNum
        matchedMs2 = matchedMs2[:, :, :, 1]
        # batch * peaksNum
        Spectrum = Spectrum[:, :, 1]
        Spectrum = torch.unsqueeze(Spectrum, dim=1)
        matchedMs2 = self.ms_encoder(matchedMs2)
        matchedMs2 = self.ms_trans_encoder(matchedMs2) + matchedMs2
        Spectrum = self.pep_encoder(Spectrum)
        Spectrum = self.pep_trans_encoder(Spectrum) + Spectrum
        x = self.consolidate(matchedMs2, Spectrum)
        x = self.general_encoder(x) + x
        x = self.encoder(x) + x
        # mean pooling, batch * d_out
        x = torch.mean(x, dim=1)
        out = self.decoder(x)
        return out


class LogCosh(nn.Module):
    def __init__(self) -> None:
        super(LogCosh, self).__init__()

    def forward(self, y, y_pred):
        loss = torch.log(torch.cosh(y - y_pred))
        return loss.sum()


class SMAE(nn.Module):
    def __init__(self, epsilon: float = 0.05) -> None:
        super(SMAE, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        square = torch.square(torch.log2(y_pred) -
                              torch.log2(y)) + self.epsilon
        return torch.mean(torch.sqrt(square))


class QuantModel(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_head: int = 4,
        dim_feedforward: int = 256,
        dropout=0.1,
        num_encoder_layers: int = 6,
        calculate_method=CALCULATE_METHOD.SUB
    ) -> None:
        super(QuantModel, self).__init__()
        self.calculate_method = calculate_method

        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_out,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.ms_trans_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )

        self.pep_trans_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_encoder_layers
        )

        self.ms_encoder = nn.Sequential(
            nn.LayerNorm(6, ),
            nn.Linear(d_in, d_out),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        self.pep_encoder = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        self.general_encoder = nn.Sequential(
            TransformerEncoder(
                encoder_layer=self.encoder_layer,
                num_layers=num_encoder_layers
            ),
            nn.SELU()
        )

        self.decoder = QuantMLP(d_out)

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def merge(x, y):
        return torch.concat((x, y), dim=1)

    @staticmethod
    def mul(x, y):
        return torch.mul(x, y)

    @staticmethod
    def sub(x, y):
        return x - y

    def consolidate(self, matchedMs2, Spectrum):
        if self.calculate_method == CALCULATE_METHOD.ADD:
            x = self.add(matchedMs2, Spectrum)
        elif self.calculate_method == CALCULATE_METHOD.SUB:
            x = self.sub(matchedMs2, Spectrum)
        elif self.calculate_method == CALCULATE_METHOD.MUL:
            x = self.mul(matchedMs2, Spectrum)
        elif self.calculate_method == CALCULATE_METHOD.MERGE:
            x = self.merge(matchedMs2, Spectrum)
        return x

    def forward(self, matchedMs2, Spectrum):
        # batch * matchedNum * peaksNum
        matchedMs2 = matchedMs2[:, :, :, 1]
        # batch * peaksNum
        Spectrum = Spectrum[:, :, :, 1]
        # Spectrum = torch.unsqueeze(Spectrum, dim=1)
        # print(matchedMs2.shape, Spectrum.shape)
        matchedMs2 = self.ms_encoder(matchedMs2)
        matchedMs2 = self.ms_trans_encoder(matchedMs2) + matchedMs2
        Spectrum = self.pep_encoder(Spectrum)
        Spectrum = self.pep_trans_encoder(Spectrum) + Spectrum
        # print(matchedMs2.shape, Spectrum.shape)
        x = self.consolidate(matchedMs2, Spectrum)
        x = self.general_encoder(x) + x
        # mean pooling, batch * d_out
        # x = torch.max(x, dim=1)[0]
        x = torch.mean(x, dim=1)
        out = self.decoder(x)
        return out


class TransformerQuantModel(nn.Module):
    def __init__(
        self,
        d_in: int = 6,
        d_out: int = 256,
        n_head: int = 4,
        num_decoder_layers: int = 6,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ) -> None:
        super(TransformerQuantModel, self).__init__()
        self.spectrum_inMLP = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        self.matchedMs2_inMLP = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SELU(),
            nn.Dropout(dropout),
        )

        self.transformer = Transformer(
            d_model=d_out,
            nhead=n_head,
            num_decoder_layers=num_decoder_layers,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.OutMLP = QuantMLP(d_out=d_out)

    def forward(self, matchedMs2: torch.Tensor, Spectrum: torch.Tensor):
        matchedMs2 = matchedMs2[:, :, :, 1]
        Spectrum = Spectrum[:, :, :, 1]
        matchedMs2 = self.matchedMs2_inMLP(matchedMs2)
        Spectrum = self.spectrum_inMLP(Spectrum)
        x = self.transformer(matchedMs2, Spectrum)
        x = torch.mean(x, dim=1)
        x = self.OutMLP(x)
        return x


class ConvQuantModel(nn.Module):
    def __init__(self) -> None:
        super(ConvQuantModel, self).__init__()
        self.activation = nn.SELU()
        self.msConvBlock = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=1),
            self.activation,
            nn.Conv1d(64, 128, kernel_size=1),
            self.activation,
            nn.Conv1d(128, 256, kernel_size=1),
            nn.Dropout(0.1)
        )

        self.peptideConvBlock = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            self.activation,
            nn.Conv1d(64, 128, kernel_size=1),
            self.activation,
            nn.Conv1d(128, 256, kernel_size=1),
            nn.Dropout(0.1)
        )

        self.mergeConvBlock = nn.Sequential(
            nn.Conv1d(256 * 2, 256, kernel_size=1),
            self.activation,
            nn.Dropout(0.1)
        )

        self.tranConvBlock = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            self.activation,
            nn.Dropout(0.1),
        )

        self.transBlocks = self._get_clones(self.tranConvBlock, 12)

        self.linearBlock = nn.Sequential(
            nn.Linear(256 * 6, 256),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, matchedMs2: torch.Tensor, Spectrum: torch.Tensor):
        matchedMs2 = matchedMs2[:, :, :, 1]
        Spectrum = Spectrum[:, :, :, 1]
        matchedMs2 = self.msConvBlock(matchedMs2)
        Spectrum = self.peptideConvBlock(Spectrum)
        x = self.mergeConvBlock(torch.concat((matchedMs2, Spectrum), dim=1))
        for i in range(len(self.transBlocks)):
            x = self.activation(self.transBlocks[i](x) + x)
        x = x.view(x.shape[0], -1)
        x = self.linearBlock(x)
        return x
    # 41.01
>>>>>>> 2fe29bdfbe3d9d607494de63d842bd48bbecce79
