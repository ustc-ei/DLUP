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