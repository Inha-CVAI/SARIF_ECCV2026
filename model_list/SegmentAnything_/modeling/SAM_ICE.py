import torch
import torch.nn as nn
import torch.fft as fft

class EdgeArtifactExtractor(nn.Module):
    def __init__(self,in_channels=6):
        super(EdgeArtifactExtractor,self).__init__()
        self.conv = nn.Conv2d(in_channels = 6,out_channels = 64,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.split = nn.Conv2d(in_channels = 64,out_channels = 6,kernel_size =1)

    def forward(self,x):
        fft_result = fft.fft2(x,dim=(-2,-1))
        real, image = fft_result.real, fft_result.imag
        fft_feature = torch.cat([real,image],dim=1)

        x = self.relu(self.bn(self.conv(fft_feature)))
        split_feature = self.split(x)

        real_part, imag_part = split_feature[:,:3,:,:], split_feature[:,3:,:,:]
        real_part = real_part.to(torch.float32)  # 실수 부분을 float32로 변환
        imag_part = imag_part.to(torch.float32)  # 허수 부분을 float32로 변환
        edge_features = fft.ifft2(torch.complex(real_part, imag_part), dim=(-2, -1))

        return edge_features.real
def extract_dct_coefficients(image):
        dct_coefficients = fft.fft2(image, dim=(-2, -1))
        return dct_coefficients.real

def extract_quantization_table(image):
        batch_size, _, height, width = image.size()
        quant_table = torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 8, 8)
        quant_table = quant_table.repeat(batch_size, 1, 1, 1)  # Repeat for batch size
        return quant_table

class CompressionArtifactExtractor(nn.Module):
    def __init__(self):
        super(CompressionArtifactExtractor,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    
    def forward(self,image):
        dct_features = extract_dct_coefficients(image)

        quant_table = extract_quantization_table(image)

        T = 1  # Threshold value
        binarized = torch.clip(dct_features, -T, T)
        binarized = (binarized != 0).float()

        # Simulate compression artifacts using quantization table
        repeated_quant_table = quant_table.repeat(1, 1, dct_features.size(-2) // 8, dct_features.size(-1) // 8)
        repeated_quant_table = repeated_quant_table.expand(-1, -1, dct_features.size(-2), dct_features.size(-1))
        repeated_quant_table = repeated_quant_table.to('cuda')
        # Convolutional processing
        x = self.relu(self.bn1(self.conv1(binarized)))
        x = self.relu(self.bn2(self.conv2(x)))

        repeated_quant_table = x*repeated_quant_table
        x = torch.cat([x,repeated_quant_table],dim=1) # 6
        
        return x
    
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9*256*256, 128),
            nn.ReLU(),
            nn.Linear(128, 768)
        )
        self.EAE = EdgeArtifactExtractor()
        self.CAE = CompressionArtifactExtractor()

    def forward(self, x):
        # Concatenate features and apply MLP
        edge_features = self.EAE(x) # 3
        compression_features = self.CAE(x) # 6

        fused_features = torch.cat([edge_features, compression_features], dim=1) # 131
        batch_size,c,h,w= fused_features.shape
        fused_features = fused_features.view(batch_size,-1)

        return self.mlp(fused_features)

# edge_extractor = EdgeArtifactExtractor()
# compression_extractor = CompressionArtifactExtractor()
# feature_fusion = FeatureFusion()

# input_image = torch.randn(1, 3, 256, 256)

# print("Testing EdgeArtifactExtractor...")
# edge_features = edge_extractor(input_image)
# print(f"EdgeArtifactExtractor output shape: {edge_features.shape}")

# print("Testing CompressionArtifactExtractor...")
# compression_features = compression_extractor(input_image)
# print(f"CompressionArtifactExtractor output shape: {compression_features.shape}")

# print("Testing FeatureFusion...")
# fused_output = feature_fusion(input_image)
# print(f"FeatureFusion output shape: {fused_output.shape}")
