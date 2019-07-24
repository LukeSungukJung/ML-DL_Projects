import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, embed_num, class_num, ):
        super(CNN_Text, self).__init__()
        
        V = embed_num
        D = 170 #args.embed_dim
        C = class_num
        Ci = 1
        Co = 40 #args.kernel_num
        Ks = [1,2,3]

        self.embed = nn.Embedding(V, D)  # 입력문장 내 43개 단어의 word vector에 random 초기값

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(len(Ks)*Co, C)  # (60, 2)

    def forward(self, x):
        # x : 패턴 파일의 한 줄(한 문장)
        #print (x.size())
        x = self.embed(x)  # (N, W, D) (1문장, 패턴의 단어수, 100차원)
        # 문장 x의 단어 벡터값 가져옴
        #print (x.size())
        x = x.unsqueeze(1)  # (N, Ci, W, D) # Ci를 추가하여 unsqueeze (Text는 채널 1개, 이미지는 채널 3개)
        #print (x.size())
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # 컨볼루션 하는 부분. 입력문 별로 호출. convs1에 필터 3개 있음 (각 1, 2, 3개 단어)
        # 입력문장에 필터 3개를 적용
        #print(len(x))
        #print(x[0].size())
        #print(x[1].size())
        #print(x[2].size())
        
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # max pooling해서 60개 값이 나옴
        
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C) 
        # 결과값(logit)은 실수값 2개 출력, 이 중 큰값을 가진 쪽이 분류된 클래스
        #print(logit)
        return logit
