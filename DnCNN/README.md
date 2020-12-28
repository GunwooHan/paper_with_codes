# DnCNN
---
# [Beyond a Gaussian Denoiser : Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/pdf/1608.03981.pdf)
---

# 소개
---
- VGG16 기반으로 작성함

<br>
<br>
<br>

# 성능
---
## 논문 제시
---
### DnCNN-S
|데이터셋|sigma 15|sigma 25|sigma 50|
|:---:|:---:|:---:|:---:|
|Set12||||
|BSD68|31.73|29.23|26.23 |

<br>
<br>

### DnCNN-S
|데이터셋|sigma 15|sigma 25|sigma 50|
|:---:|:---:|:---:|:---:|
|Set12||||
|BSD68|31.73|29.23|26.23 |

<br>
<br>

## 내 코드
---
|데이터셋|sigma 15|sigma 25|sigma 50|
|:---:|:---:|:---:|:---:|
|Set12|30.7294|||
|BSD68|29.62|||

<br>
<br>
<br>

# Requirements
---
- tensorflow 2.3.1 (2.1<=) 
- numpy 1.18.5 (1.18.5<=) 
- matplotlib 3.3.2
- albumentations 0.5.2


<br>
<br>
<br>

# Future work
---
- 성능향상
- input data Normalization 
- requirements.txt 작성