> 벨로그에서 이 글을 비공개 해버리길래 깃허브로 일단 옮겨왔습니다. 무악재에는 코드 올리는 법을 몰라서 우선은 여기에 올린다음 천천히 수정합니다. :)

왜 갑자기 여기에 꽂혔는지는 모르겠으나, CeVIO나 SynthV같은, 아니 적어도 우타우 같은 음성 합성 엔진 및 라이브러리 제조를 저도 해보고 싶어졌습니다.

딥보컬 쓰면 끝나는 건데, 사실 음합엔은 AI들어간 것밖에 안 써봐서, 과연 온라인에 공개되어있는 코드만 보고 AI 음합엔을 단 며칠만에 얼마나 재현해낼 수 있나 테스트를 해보았습니다.

# 0. 참고
우선 이 포스트는 [Tensorflow 공식 사이트의 Pix2Pix 자료](https://www.tensorflow.org/tutorials/generative/pix2pix?hl=ko)를 참조했음을 밝힙니다. 그리고 STFT 관련해서 도움 주신 서림님께도 감사드립니다.

# 1. 데이터셋
## 결과 데이터(1.mp3, 2.mp3, ...)
wav보다는 mp3를 추천드립니다. 사유는 그냥 용량이 작아요. 샘플 레이트도 뭔가 일반적이지는 않은데, 제가 내보낸 파일들은 이상하게 다 22050이었습니다(일반적으로는 44100, 48000).

샘플 레이트가 다르다면 그걸 반영해서 코드를 짜시면 되는데, 어차피 STFT를 적용하고 리사이즈 하는 과정에서 크기가 통일될 겁니다. 파일 길이는 **10초**. 무조건 10초 하시면 됩니다. 10초 안되면 길이를 늘려서 맞춰주시고, 10초가 넘어가는 파일은 중간의 공백 타이밍에 끊어주시거나 그게 안된다면 과감히 버려주세요.

데이터셋을 처음부터 직접 만들어야 하다 보니 아무래도 힘이 좀 들 것 같습니다. 솔직히 우타우나 딥보컬 만드는 게 훨씬 쉽습니다. 저는 아직 3개의 데이터셋 밖에 없지만 적어도 200개 이상을 목표로 하고 있습니다. 그리고 그 200개의 데이터셋 안에 필요한 모든 음소가 들어가 있어야 하고(이것 때문에 일본어를 추천해요), 다양한 음을 다양한 상황에서 내는 데이터라면 더 좋겠습니다.

그리고 남의 목소리 배경음 빼고 사용하지 마십시오. 남의 목소리 쓰는 우타우 만들면 큰일나듯이, 이것도 마찬가지 입니다.

## 입력 데이터(.txt)
CSV가 아니고 리스트입니다. 3차원 배열이기 때문에 엑셀로 만들기 어렵습니다. 이 리스트의 각 원소는,

```
[
    [ XX.XX, YY, 'ZZ' ], 
    [ XX.XX, YY, 'ZZ' ], 
    [ XX.XX, YY, 'ZZ' ],
    ..., 
    [ XX.XX, YY, 'ZZ' ]
]
```

형식으로 저장되어 있습니다. (이게 한 개의 데이터이고, 이런 데이터가 여러 개가 필요합니다.) 

여기에서 `XX.XX` 는 로직에서의 재생헤드 위치입니다. 딱히 밀리초 같은 단위가 아니었습니다 정확히 말하면 `30.00`이 1초였어요.

`YY`는 피치입니다. 0을 A2로 잡았지요. 아무 음도 내지 않을 때도 그냥 0을 입력하시면 됩니다. 어차피 뒤에 설명할 전처리 과정에서 잘리거든요.

`'ZZ'` 는 음가입니다. 문자열로 저장했구요, 아마 SynthV의... 발음기호를 따르고 있던 걸로 기억합니다. 참고로 음절 단위가 아니라 음소 단위로 입력하셔야 합니다. 음절 단위는 너무 데이터가 복잡해지더라구요.

그러니까 이제, 재생헤드 위치 30.02부터 45.0까지 A2음으로 'ra'를 낸다고 하고, 30.02부터 33.0 정도가 'r', 33.0부터 45.0까지가 'a'라고 가정한다면,

```[[ 0, -12, '' ], [ 30.02, 0, 'r'], [ 33, 0, 'a'], [ 45, -12, '' ]]```

이런 배열을 저장하시면 되는 겁니다. 왜 앞뒤에 빈 문자열이 붙어 있냐하면, 그 위치에서는 아무 소리도 나지 않는 공백일 것이므로, 그 공백을 표시해 주기 위해서입니다.


# 2. 코드

## 2.1 필요한 라이브러리 임포트, 함수 정의

```python
# python 표준 라이브러리
import os
import pathlib
import time
import datetime

# 오픈소스 라이브러리
from matplotlib import pyplot as plt
from IPython import display
import tensorflow as tf
from tensorflow.python.keras import layers
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL
import librosa

# Colab Notebook Library
import IPython.display as ipd

num = {'' : 1, 'a': 0.12, 'e': 0.14, 'i': 0.16, 'o': 0.18, 'u': 0.2, 'y':0.25,
            'h': 0.3, 'r': 0.4, 'n': 0.5, 'N': 0.52, 'm': 0.54,
            'b': 0.6, 'd': 0.62, 'g': 0.64,
            'p': 0.7, 't': 0.72, 'T': 0.74, 'k': 0.76, 
            'ch': 0.8, 'ts': 0.82, 'z': 0.84, 's': 0.86, 'sh': 0.88}

freqArray = np.multiply([110, 117, 123, 131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262], 1/5)
freqOneHot = tf.one_hot(freqArray, 1000)
for i in range(12):
  freqOneHot = tf.math.add(freqOneHot, tf.one_hot(np.multiply(freqArray, i+2), 1000, on_value=0.8))
```
  
이 중 `num`은 음소에 따른 가중치이고(정작 곱하는 것은 `1-num`이 되지만요), `freqOneHot`은 주파수에 따른 원-핫 인코딩입니다.

이 배열들을 바탕으로 앞에서 불러온 입력 데이터를 이미지의 형태로 바꾸어 줄 것입니다. 그런데 어떤 이미지일까요?

STFT라는 변환이 있습니다. 아래 이미지의 왼쪽 그림과 같은 오디오 배열을 넣고 돌리면 오른쪽 그림과 같은 데이터로 변환해 줍니다. (저는 여기에서 librosa의 stft를 사용하고 있습니다. 그것이 제일 편리하고, 값이 잘 나오더군요)

![](https://velog.velcdn.com/images/hyun1008/post/7947e99d-d649-4279-9cfc-b97e0d5848cd/image.png)

여기에서 데이터의 y축은 주파수 값과 관련이 있습니다. 그 말은, 입력 데이터의 음정 값을 이용해서 오른쪽과 같은 이미지와 최대한 근접한 것을 만들어낼 수 있다는 것이 아니겠습니까?

오디오의 STFT 분석 값은 시간마다 1차원 배열이 존재하는 형태입니다. 그래서 우리도 각 순간마다 1차원 배열을 만들어 줄 것입니다.

아래 이미지를 볼까요?

![](https://velog.velcdn.com/images/hyun1008/post/997b4dc4-f799-4864-83e0-789cd87e27fc/image.jpeg)

처음 불러온 입력 데이터는 왼쪽과 같이 되어 있었습니다. 후술할 전처리 과정을 통해서, 입력 데이터를 오른쪽과 같이 변환해서, 각 순간마다의 1차원 배열을 만들어 주어야 합니다. 그리고 그 사이즈가 STFT를 적용했을 때의 결과 데이터와 같아야죠. 그래야 pix2pix를 쓸 수 있거든요.

기본적으로 사용할 것은 **원핫 인코딩** 입니다. 만약 A음을 원한다면, A=440Hz이므로 주파수 440에서 1을 찍어 주고 나머지는 다 0이 들어간 행렬을 반환하면 되는 겁니다.

그런데.. 세상일이 그렇게 쉽지만은 않습니다.

음파는 여러 주파수를 갖는 파동의 조합이고, 만약 A음을 낸다 한다면 A의 2배음, 3배음, ... 등등에 해당하는 파동이 조금씩 들어가게 됩니다. 그래서 위의 STFT 그래프가 마치 밀푀유처럼 여러 선이 겹쳐져 보이는 것입니다.

그러니, 제가 A음을 원한다면, 110Hz, 220Hz, 440Hz, 880Hz .... 등의 주파수에 모두 값을 넣어 주어야 제대로 된 입력 데이터가 나온다는 것입니다.

바로 이런 입력 데이터가 말이죠.

![](https://velog.velcdn.com/images/hyun1008/post/aaca4ba5-c123-4619-9a11-82e83dc187b1/image.png)

(짜잔)

그러므로, 우리는 우선 각 음에 맞는 주파수를 찍어 줄 것입니다. 인덱스 0이 A2 입니다. 이 Array를 5.6으로 나누어 준 것은 STFT의 결과값 스케일이랑 맞추기 위함이고 별다른 이유는 없습니다..

```python
freqArray = np.multiply([110, 117, 123, 131, 139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262], 1/5.6)
```

그리고, 이 배열을 원-핫 인코딩 해 줍니다.

```python
freqOneHot = tf.one_hot(freqArray, 1000)
```

배열의 12배까지 가중치 0.8을 주어 원-핫 인코딩 된 배열에 더해 줍니다.

```python
for i in range(12):
  freqOneHot = tf.math.add(freqOneHot, tf.one_hot(np.multiply(freqArray, i+2), 1000, on_value=0.8))
```

그러면 각 인덱스에서 다음과 같은 결과값이 출력됩니다.

```
tf.Tensor(
[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.8 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.8 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.8 0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ], shape=(1000,), dtype=float32)
```

각 음에 따라서 이 배열을 시간에 맞추어 넣어주면 된다는 것입니다.


## 2.2 데이터들 가져오기

### 2.2.1 입력 데이터

```python
#@title 배열 형태로 저장해주세요. { display-mode: "form" }

fileName = "input_data.txt" #@param {type:"string"}
input_data = open(fileName, "r")
dataArray = eval(input_data.read())
```

저는 Colab 환경에서 수행했기 때문에 Colab에 맞는 마크다운들이 조금 들어가 있습니다.

아무튼 파일을 가져와서 dataArray라는 배열로 저장해주었습니다.

### 2.2.2 결과 데이터

```python
#@title 준비된 데이터셋의 개수 { display-mode: "form" }

N_DataSet = 3 #@param {type:"number"}

y = []
sr = []
Zxx = []

for i in range(N_DataSet):
  y.append(librosa.load(str(i)+'.mp3')[0])
  sr.append(librosa.load(str(i)+'.mp3')[1])
  Zxx.append(librosa.stft(y[i], n_fft=500))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(y[0])

plt.subplot(1, 2, 2)
plt.pcolormesh(abs(Zxx[0]), vmin=0, vmax=1)
plt.show()
```

네 제 데이터셋은.. 3개...초라합니다. 하지만 조만간 많이 추가할 계획입니다... 준비된 데이터의 수만큼 N_DataSet을 늘려주시고 for문으로 불러온다음 stft도 수행해 줍니다.

## 2.3 데이터셋의 형태로 변경

```python
train_set = []
for i in range(len(dataArray)):
    train_set.append([])
    for k in range(1770):
      for l in range(len(dataArray[i])):
        if l == len(dataArray[i]) - 1:
          if k >= dataArray[i][l][0] * 2 / 60 * 177 and k < 1770:
            train_set[i].append([(freqOneHot[dataArray[i][l][1]]*(1-num[dataArray[i][l][2]]))])
        else:
          if k >= dataArray[i][l][0] * 2 / 60 * 177 and k < dataArray[i][l+1][0] * 2 / 60 * 177:
            train_set[i].append([(freqOneHot[dataArray[i][l][1]]*(1-num[dataArray[i][l][2]]))])
```

앞에서 말했던 입력 데이터를 시간마다의 배열로 전처리하는 과정입니다.

앞에서 1765개의 시간마다의 데이터가 나왔습니다만(1초당 176.5개), 일단 당장은 1770으로 짠 후 리사이즈하려고 합니다. 그래야 1초당 데이터의 갯수가 정수가 나오니까요.

그리고 여기에서 가중치 `1-num`을 곱해주었는데요, 이것은 앞에서도 말씀드렸다시피 음소에 따른 가중치입니다. 녹음을 많이 해보면, `a` 같은 모음에서는 매우 규칙적인 파동이 발생하고 크기도 크며, `g` 같은 자음에서는 규칙적인 파동이 아니라 파열음 위주로 나오며 음의 크기도 작다는 것을 알 수 있어요. 그래서 이 특징들을 반영해서 값을 바꾸어 넣어주었습니다.

사실 이것도 원핫인코딩 해버리면 좋을지도 모르겠지만, pix2pix 원본에서도 RGB 세 개의 채널의 값만을 가지고 창문이라든지 테라스 같은 것을 구현해 냈으니, 저에게도 좋은 결과 있기를 바라야죠.

```python
y_tensor = torch.tensor(Zxx)
y_tensor_real = tf.convert_to_tensor(torch.tensor(y_tensor).real)
y_tensor_img = tf.convert_to_tensor(torch.tensor(y_tensor).imag)

y_tensor_real = tf.expand_dims(y_tensor_real.numpy(), 3)
y_tensor_img = tf.expand_dims(y_tensor_img.numpy(), 3)

x_tensor = []
for i in range(len(Zxx)):
  x_tensor.append(tf.transpose(tf.squeeze(tf.convert_to_tensor(train_set[i]), 1)))
x_tensor = tf.image.resize(tf.expand_dims(tf.convert_to_tensor(x_tensor), 3), [1000, 1765])

y_tensor = tf.math.multiply(tf.concat([y_tensor_real,y_tensor_img], axis=1), 2)
trainTensor = tf.concat([x_tensor,y_tensor], axis=1)

def load_image_train(input):
  x = input[:1000]
  y = input[1000:]
  return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

def load_image_test(input):
  x = input[:1000]
  y = input[1000:]
  return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

# The facade training set consist of 400 images
BUFFER_SIZE = N_DataSet
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1

OUTPUT_CHANNELS = 1

train_dataset = tf.data.Dataset.from_tensor_slices(trainTensor)
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(trainTensor)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)
```
그다음은 다음과 같은 작업을 통해 데이터셋을 최종적으로 만들어 주었습니다.

* `y_tensor_real`, `y_tensor_img`: 복소수 형태로 저장되어 있는 STFT의 결과물을 실수부와 허수부로 분리해 줍니다. 그리고 이걸 합쳐서 `y_tensor`에 다시 저장해줍니다.

* `x_tensor`: (1770, 1000)으로 차원이 뒤집혀있는 것을 transpose 하여 (1000, 1770)으로 맞추어 주고 `y_tensor`와 모양을 맞춥니다.

* `trainTensor`: `x_tensor`과 `y_tensor`의 결합입니다. 이제 `trainTensor`를 보시면, 각 데이터의 처음 1000줄은 입력 데이터가 들어가구요, 다음 251줄은 결과 데이터의 실수부가, 마지막 251줄은 결과 데이터의 허수부가 들어갑니다.

* `load_image_train`, `load_image_test`: 합쳐져있는 `trainTensor`를 데이터셋으로 저장할 때, 이걸 슬라이스해주는 함수입니다. 이 과정이 필요한 이유는 (원본을 따라했을 뿐이기 때문에) 잘 모르겠지만 아마 같은 입출력값의 매핑이 아닐까 싶습니다.

* `train_dataset`, `test_dataset`: 학습용 데이터셋과 결과 출력용 데이터셋입니다.. 저는 양이 적다보니 굳이 안나누고 같은 셋으로 수행을 했습니다(이러면안됨).

## 2.4 모델의 정의
### 2.4.1 생성기

```python

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()

  result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result
  
```
```python
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
```

```python
def Generator():

  inp = tf.keras.layers.Input(shape=[1000, 1765, 1])
  x = tf.image.resize(inp, [512, 1024], antialias=True)
  x = tf.keras.layers.concatenate([x, x], 1) # size=(1024, 1024)

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 512, 512, 64)
    downsample(128, 4),  # (batch_size, 256, 256, 128)
    downsample(256, 4),  # (batch_size, 128, 128, 256)
    downsample(512, 4),  # (batch_size, 64, 64, 512)
    downsample(512, 4),  # (batch_size, 32, 32, 512)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 16, 16, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 32, 32, 1024)
    upsample(512, 4),  # (batch_size, 64, 64, 1024)
    upsample(256, 4),  # (batch_size, 128, 128, 512)
    upsample(128, 4),  # (batch_size, 256, 256, 256)
    upsample(64, 4),  # (batch_size, 512, 512, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 1024, 1024, 1)


  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)
  x = tf.image.resize(x, [502, 1765], antialias=True)

  return tf.keras.Model(inputs=inp, outputs=x)
```

원본의 코드를 그대로 따라갔습니다만 처음 정사각형 텐서로 변환해주는 과정에서 원본보다 4배 큰 1024x1024를 채택했습니다.

```python
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
```

이렇게 하면 다음과 같이 어마무시한 크기의 모델이 그려집니다.

![](https://velog.velcdn.com/images/hyun1008/post/b1271053-ac47-4c8c-9f22-584d3090978b/image.png)

```python
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(tf.image.resize(target, [502, 1765]) - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
```

로스는 이렇게 정의되었는데요. 이것도 원본과 같습니다.

### 2.4.2 판별기

```python
def downsample2(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.9)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv1D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result
```

```python
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.2)

  inp = tf.keras.layers.Input(shape=[1000, 1765, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[None, 1765, 1], name='target_image')

  inp2 = tf.image.resize(inp, [128, 883], antialias=True)
  tar2 = tf.image.resize(tar, [256, 883], antialias=True)
  inp3 = tf.keras.layers.concatenate([inp2, inp2], 1) 
  x = tf.keras.layers.concatenate([inp3, tar2]) 

  down1 = downsample2(64, 2, False)(x)  # (batch_size, 256, 442, 64)
  down2 = downsample2(128, 2)(down1)  # (batch_size, 256, 221, 128)
  down3 = downsample2(128, 2)(down2)  # (batch_size, 256, 111, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(16, 1, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 258, 113, 16)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 260, 115, 16)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 257, 112, 1)
  
  last = tf.image.resize(last, [431, 128])

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
```

판별기에서 수정한 것은 initializer의 수치와, `Conv2D`를 `Conv1D` 로 바꿔준 것(세로축을 줄이고 싶지 않았거든요), 그리고 몇 가지 수치들입니다. 이것은 계속 바꾸어보면서 로스를 체크해보는 게 좋을 것 같아요!(아직도 미완성)

```python
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
```

![](https://velog.velcdn.com/images/hyun1008/post/e0504361-97da-4d31-88eb-3a628d1af201/image.png)

```python
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
```

판별기 로스 역시 원본을 그대로 따라갑니다.

### 2.4.3 옵티마이저

```python
generator_optimizer = tf.keras.optimizers.legacy.Adam(2e-5, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(2e-5, beta_1=0.5)
```

옵티마이저는 한 자리 내렸습니다.

## 2.5 이미지 생성기 정의


```python
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 5))
  plt.subplot(1, 2, 1)
  r1 = tf.squeeze(tf.squeeze(tf.image.resize(tar.numpy(), [502, 1765]), 3), 0)
  plt.pcolormesh(np.abs(r1), vmin=0, vmax=1)

  plt.subplot(1, 2, 2)   
  r2 = tf.squeeze(tf.image.resize(prediction[0].numpy(), [502, 1765]), 2)
  plt.pcolormesh(np.abs(r2), vmin=0, vmax=1)
  plt.show()
```
학습 중간중간에 이미지를 확인할 수 있는 이미지 생성기 입니다. 왼쪽에는 출력해야 했을 결과의 STFT 데이터가, 오른쪽에는 현재 모델이 출력할 수 있는 최선의 STFT 데이터가 그려집니다.

```python
for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)
```
![](https://velog.velcdn.com/images/hyun1008/post/e4af645d-c4fc-4ce0-822b-f346c5a0ee6d/image.png)

## 2.6 학습

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```

학습 중간 중간 체크포인트를 만들어 모델을 저장하게 되는데, 이 모델이 어디에 저장될지 지정해주는 코드입니다. 역시 건드리지는 않았습니다.

```python
#@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_total_loss = 0
    gen_gan_loss = 0 
    gen_l1_loss = 0
    disc_loss = 0
    
    for i in range(len(input_image)):
      gen_output = generator(input_image, training=True)

      disc_real_output = discriminator([input_image, target], training=True)
      disc_generated_output = discriminator([input_image, gen_output], training=True)

      gtl, ggl, gll = generator_loss(disc_generated_output, gen_output, target)
      dl = discriminator_loss(disc_real_output, disc_generated_output)

      gen_total_loss += gtl
      gen_gan_loss += ggl
      gen_l1_loss += gll
      disc_loss += dl
    
    gen_total_loss = gen_total_loss / len(input_image)
    gen_gan_loss = gen_gan_loss / len(input_image)
    gen_l1_loss = gen_l1_loss / len(input_image)
    disc_loss = disc_loss / len(input_image)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
    tf.summary.scalar('disc_loss', disc_loss, step=step)
    
```

이건 좀 건드리긴 했는데 아직은 오류없이 돌아가고 있어서 그냥 방치되어있습니다(...) 아마 `input_image`의 사이즈에 따른 조정이어서, 배치사이즈가 커지면 여기에서 오류가 날 수도 있습니다.

다만 배치사이즈는 1로 하는 것이 가장 효과가 좋다고 합니다...

```python
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 10 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 10 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step/1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 1 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 100 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
```

이제 거의 다 왔습니다. 여기에서도 좀 변경된 점이 있다면, 원래는 step 100마다 점을 찍고, 1000마다 이미지를 출력하고, 5000마다 체크포인트를 저장했었다면, 제 코드에서는 step 1마다 점을 찍고, 10마다 이미지를 출력하고, 100마다 체크포인트를 저장합니다. ~~이게 다 성격이 급한 탓입니다~~

```python
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

%reload_ext tensorboard
%tensorboard --logdir {log_dir}
```

텐서보드를 로드합니다. 이거 있으면 로스가 자동으로 그래프에 그려집니다. 아주 편해요!

```python
fit(train_dataset, test_dataset, steps=40000)
```

그리고 fit을 실행하면 학습이 진행됩니다.
저는 40000까지는 안했고, 한 30분 정도 돌렸더니 아래와 같은 정도의 그림이 나왔습니다. 아마 데이터셋이 커지면 이런 출력물을 뽑는 데에 더 오래 걸리겠죠....(진짜 4만번 돌려야 하나...)(솔직히 지금은 과적합 상태 입니다)

![](https://velog.velcdn.com/images/hyun1008/post/66d06ab0-51b5-4508-b212-509a8e9452f8/image.png)

## 2.7 테스트

```python
for inp, tar in test_dataset.take(1):
  prediction = generator(inp, training=True)
result_comp = tf.squeeze(tf.complex(tf.image.resize(prediction[0][:251], [251, 1765]), tf.image.resize(prediction[0][500:], [251, 1765])), 2).numpy()

result_comp [ np.abs(result_comp) < 0.05] = 0
for i in range(7):
  result_comp[i] = 0
plt.pcolormesh(np.abs(result_comp), vmin=0, vmax=1)
```

`generator` 로 결과물을 뽑아내 `prediction` 에 저장했습니다. 그리고 따로따로 저장되어 있던 실수부와 허수부를 모아 `result_comp`에 저장해주었습니다.

그 후 여기에 다양한 후가공을 가해주었는데요,
```python
result_comp [ np.abs(result_comp) < 0.05] = 0
```
먼저 절댓값이 0.05 미만인 값은 잘라주었습니다.

```python
for i in range(7):
  result_comp[i] = 0
```

그리고 아래서부터 5-7줄정도 너무 낮은 주파수를 가지고 있는 성분도 0으로 잘라주었습니다(대충.. 로우컷이라는 개념이죠.)

![](https://velog.velcdn.com/images/hyun1008/post/2e817a49-095f-4acd-b221-cb98d846616a/image.png)

기분탓이겠지만, 조금 더 선명하게 보입니다.

```python
result_ifft = librosa.istft(result_comp, n_fft=500)
print(result_ifft)
plt.ylim(-1, 1)
plt.plot(result_ifft)
```

마지막으로 이 `result_comp` 값에 `librosa.istft` 를 걸어주면 딥러닝을 통해 구현된 오디오 파일이 나옵니다.

![](https://velog.velcdn.com/images/hyun1008/post/f3882c6d-1524-4a4a-b782-dfbcf64b84b8/image.png)

```python
from scipy.io.wavfile import write

rate = 22050
ipd.Audio(result_ifft, rate=rate)
write('test.wav', rate, result_ifft)
```

결과물을 `.wav` 파일로 저장할 수 있습니다.

[![Video Label](http://img.youtube.com/vi/MYRuvjNkam4/0.jpg)](https://youtu.be/MYRuvjNkam4?t=0s)

잡음이 굉장히 많이 끼어 있지만, 학습이 얼마 진행되지 않아 그렇다고 믿고 넘기려고 합니다.

그럼 여기에서 마치도록 하겠습니다!
데이터셋이 충분히 커졌을 때 결과물이 안좋게 나오면 이 글은 소리소문없이 삭제될 수 있습니다(헣허

