import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # row 5000에 col 임베딩 차원이랑 맞춰줌
        # squeeze: 차원 줄이기, unsqueeze: 특정 위치(인덱스)에 1인 차원 추가하기 -> 여기서는 0부터 5000(맥스 토큰수)까지 arange한담에 float화해서 차원을 ([0,1,2]) -> ([[0.0000e+00], [1.0000e+00], [2.0000e+00]]) 로 만듦
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 2차원으로 만든 이유는 input embedding matrix랑 element wise로 더해서 인코더에 넣으려고

        # 이 논문에서 position을 학습시키기 위해 사용한 방법임. 다른 방법으로도 position을 모델에게 알려줄 수 있으면 된 것. 성능의 큰차이는 없다고 함
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수번째 단어는 sin함수
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수번째 단어는 cos함수
        # 위에 position처럼 1 -> 2차원으로 늘려서 transpose 시킴 그러니까 col vector 되는거
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # TODO: 어디서 쓰는거임

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)  # dropout시켜서 내보냄


class Encoder(nn.Module):
    # ntoken: lenth of vocabs, ninp: embedding dimension
    def __init__(self, ntoken, ninp, dropout=0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)  # row: 토큰수, col: 임베딩차원
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):  # weight -0.1 ~ 0.1 사이 균등분포로 초기화
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)


class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):  # token수, embedding dimension
        super().__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        # permute: 차원재배열. 0,1,2차원을 1,0,2 차원의 위치로 재배열하는 함수임
        return self.decoder(inp).permute(1, 0, 2)
