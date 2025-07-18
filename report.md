### 一、**Generator设计思路**
#### **1. 输入处理**
- **输入**：100维噪声向量（`noise_dim=100`），通过`reshape`转换为`(N, 100, 1, 1)`的张量。
- **目的**：将低维噪声逐步上采样为128×128的RGB图像。

#### **2. 反卷积层（Transposed Convolution）**
- **层级设计**：6层反卷积，每层通过`stride=2`实现2倍上采样：
  ```python
  # 反卷积核配置：kernel_size=4, stride=2, padding=1
  nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
  ```
  - 从`4×4` → `8×8` → `16×16` → `32×32` → `64×64` → `128×128`。
- **通道数变化**：  
  `1024 (feature_map*16)` → `512` → `256` → `128` → `64` → `3`（RGB输出）。

#### **3. 关键组件**
- **BatchNorm**：每层反卷积后接批量归一化（除输出层外），加速训练并稳定梯度。
- **ReLU激活**：使用`ReLU`引入非线性（输出层用`Tanh`将像素值约束到`[-1, 1]`）。
- **偏置禁用**：所有反卷积层设置`bias=False`，由`BatchNorm`接管偏移量。

#### **4. 输出层**
- 最后一层反卷积将64通道特征映射到3通道RGB图像，尺寸从`64×64`上采样到`128×128`。
- `Tanh`激活函数确保输出值在`[-1, 1]`范围内，与归一化后的输入数据分布一致。

---

### **二、Discriminator设计思路**
#### **1. 输入处理**
- **输入**：128×128的RGB图像（`3×128×128`）。
- **目的**：逐步下采样并输出图像为真实（1）或生成（0）的概率。

#### **2. 卷积层（Convolution）**
- **层级设计**：7层卷积，每层通过`stride=2`实现2倍下采样：
  ```python
  # 卷积核配置：kernel_size=4, stride=2, padding=1
  nn.Conv2d(in_channels, out_channels, 4, 2, 1)
  ```
  - 从`128×128` → `64×64` → `32×32` → `16×16` → `8×8` → `4×4` → `2×2`。
- **通道数变化**：  
  `3` → `64` → `128` → `256` → `512` → `1024` → `2048 (feature_map*32)`。

#### **3. 关键组件**
- **LeakyReLU激活**：使用`LeakyReLU(0.2)`缓解梯度消失（负斜率0.2）。
- **BatchNorm**：除第一层外每层卷积后接批量归一化。
- **自适应池化**：最终通过`AdaptiveAvgPool2d(1)`将特征图压缩为`1×1`，再通过`1×1`卷积输出单通道概率。
- **Sigmoid激活**：将输出映射到`[0, 1]`，表示图像为真的概率。

#### **4. 输出层**
- 最终通过`view`将输出展平为`(batch_size, 1)`，与二元标签（0/1）匹配。


### 三、网络结构优化思路
1. **生成器(Generator)升级**：
   - 新增最后一层转置卷积：`nn.ConvTranspose2d(64, 3, 4, 2, 1)`  
     将64×64上采样到128×128
   - 保持前5层结构不变（100→1024→512→256→128→64）
   - 输出仍用Tanh激活约束到[-1,1]

2. **判别器(Discriminator)升级**：
   - 新增第一层卷积：`nn.Conv2d(3, 64, 4, 2, 1)`  
     将128×128下采样到64×64
   - 后续6层保持原下采样结构（64→128→256→512→1024→2048→1）
   - 最终通过全局平均池化输出概率

3. **对更高方便率的核心改进**：
   ```python
   # 生成器新增层（原最后层输出64x64）
   nn.ConvTranspose2d(64, 3, 4, 2, 1)  # 新增的128x128输出层

   # 判别器新增层（原第一层输入64x64）
   nn.Conv2d(3, 64, 4, 2, 1)  # 新增的128x128输入层

   # preprocess 环节将图片生成改为生成128*128

   # dataloader 和 train 环节将img_dim = 64 修改为128
   ```

### 四、**train.py的大致思想**
1. **数据准备**：
   ```python
   transform = Compose([
       Resize(128),  # 关键修改点 符合生成的 128*128图像
       CenterCrop(128),
       ToTensor(),
       Normalize((0.5,), (0.5,))
   ```

2. **对抗训练循环**：
   ```python
   for real_imgs in dataloader:
       # 1. 训练判别器
       # - 真实图片前向传播（128x128输入）
       # - 生成128x128假图片并前向传播
       # - 计算判别器损失(real_loss + fake_loss)
       
       # 2. 训练生成器
       # - 生成128x128假图片
       # - 让判别器误判
   ```

3. **具体思路**：
    -  生成器(G)：将随机噪声→逼真图像（让判别器误判）
    -  判别器(D)：区分真实图像 vs 生成图像（当"鉴定专家"）
    -  训练流程（交替优化）
      -   第一步：训练判别器（固定G）
      -   用真实图片训练D输出1
      -   用G生成的假图片训练D输出0
      -   计算D的总损失（真+假）并反向传播
    -   第二步：训练生成器（固定D）
      -   让G生成的图片骗过D（使D输出接近1）
      -    计算G的损失并反向传播
    -   交替冻结：训练D时冻结G，训练G时冻结D
    -   标签平滑：用0.9/0.1代替1/0防过拟合
    -   总体而言，就是让让生成器和判别器在对抗中共同进化，最终G能生成以假乱真的128x128图像，但是不能让其中一个训练的过快
4. **失衡表现**
    1.  判别器（D）学习过快
        - D的损失迅速下降至接近0，准确率接近100%（能完美区分真假样本）
        - G的损失居高不下或剧烈波动，生成的图片质量低（模糊、重复模式）
        - 梯度消失：D过于强大，导致传给G的梯度（误差信号）趋近于零，G无法继续优化
        - 模式崩溃（Mode Collapse）：G发现某些样本能短暂骗过D，会反复生成这些样本，导致输出多样性丧失（例如生成的人脸全是同一张）
        - 解决方法：降低D的学习率，减少D的训练频率，使用WGAN-GP等改进损失函数，避免梯度消失
    2.  生成器（G）学习过快
        - G的损失迅速下降，但D的损失持续上升
        - 生成的图片看似合理但细节怪异（如人脸五官错位），D无法提供有效反馈
        - 由于判别器失效，D无法区分真假，导致对抗博弈失去意义
        - 导致生成质量停滞，G缺乏有效对抗，生成的图片难以进一步提升真实性
        - 解决方法：增加D的容量，暂时冻结G的训练，优先强化D的鉴别能力，在损失函数中引入感知损失（Perceptual Loss），补充图像质量评估指标
### 五、**代码**
1. network.py
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_map=64, img_channels=3):
        super(Generator, self).__init__()  # 初始化生成器
        # 定义生成器的主网络结构
        # 输入: N x noise_dim x 1 x 1
        # 使用反卷积层将噪声向量转换为图像
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_map * 16, 4, 1, 0, bias=False),  # -> 4x4
            nn.BatchNorm2d(feature_map * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 16, feature_map * 8, 4, 2, 1, bias=False),  # ->8x8
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 8, feature_map * 4, 4, 2, 1, bias=False),  # -> 16x16
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 4, feature_map * 2, 4, 2, 1, bias=False),  # -> 32x32
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map * 2, feature_map, 4, 2, 1, bias=False),      # -> 64x64
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map, img_channels, 4, 2, 1, bias=False),         #-> 128x128
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)



class Discriminator(nn.Module):
    def __init__(self,feature_map=64, img_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: 3 x 128 * 128
            nn.Conv2d(img_channels, feature_map, 4, 2, 1, bias=False),   # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map, feature_map * 2, 4, 2, 1, bias=False),   # -> 32x32
            nn.BatchNorm2d(feature_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 2, feature_map * 4, 4, 2, 1, bias=False),  # -> 16x16
            nn.BatchNorm2d(feature_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 4, feature_map * 8, 4, 2, 1, bias=False),  # -> 8x8
            nn.BatchNorm2d(feature_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 8, feature_map * 16, 4, 2, 1, bias=False),  # -> 4x4
            nn.BatchNorm2d(feature_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map * 16, feature_map * 32, 4, 2, 1, bias=False),  # -> 2x2
            nn.BatchNorm2d(feature_map * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # 最后使用自适应平均池化将特征图大小调整为 1x1
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_map * 32, 1, 1, 1, 0, bias=False),  # -> 1x1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(x.size(0)) # 输出为(batch_size, 1)

```
2. preprocess.py
```python
from PIL import Image
import os.path
import os
import threading
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""多线程将图片缩放后再裁切到128*128分辨率"""
# 裁切图片宽度
w = 128  # 修改为128
# 裁切图片高度
h = 128  # 修改为128
# 裁切点横坐标(以图片左上角为原点)
x = 0
# 裁切点纵坐标
y = 20


def cutArray(l, num):
    avg = len(l) / float(num)
    o = []
    last = 0.0

    while last < len(l):
        o.append(l[int(last) : int(last + avg)])
        last += avg

    return o


def convertjpg(jpgfile, outdir, width=w, height=h):
    img = Image.open(jpgfile)
    (l, h) = img.size
    rate = min(l, h) / width  # 使用新的width=128计算缩放比例
    try:
        img = img.resize((int(l // rate), int(h // rate)), Image.BILINEAR)
        img = img.crop((x, y, width + x, height + y))
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


class thread(threading.Thread):
    def __init__(self, threadID, inpath, outpath, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.inpath = inpath
        self.outpath = outpath
        self.files = files

    def run(self):
        count = 0
        try:
            for file in self.files:
                convertjpg(self.inpath + file, self.outpath)
                count = count + 1
        except Exception as e:
            print(e)
        print(f"Image already processed: {count}")


if __name__ == "__main__":
    inpath = "img/img_align_celeba/"
    outpath = "img/processed/" 
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    files = os.listdir(inpath)
    files = cutArray(files, 8)
    T1 = thread(1, inpath, outpath, files[0])
    T2 = thread(2, inpath, outpath, files[1])
    T3 = thread(3, inpath, outpath, files[2])
    T4 = thread(4, inpath, outpath, files[3])
    T5 = thread(5, inpath, outpath, files[4])
    T6 = thread(6, inpath, outpath, files[5])
    T7 = thread(7, inpath, outpath, files[6])
    T8 = thread(8, inpath, outpath, files[7])

    T1.start()
    T2.start()
    T3.start()
    T4.start()
    T5.start()
    T6.start()
    T7.start()
    T8.start()

    T1.join()
    T2.join()
    T3.join()
    T4.join()
    T5.join()
    T6.join()
    T7.join()
    T8.join()
```
3. dataloader.py
```python
import os
import cv2
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 设置图像尺寸 
img_dim = 128 #修改为128
PATH = "img/processed/"


class DataGenerater(Dataset):
    """
    数据集定义
    """

    def __init__(self, path=PATH, transform=None):
        """
        构造函数
        """
        self.dir = path
        self.datalist = os.listdir(PATH)
        self.image_size = (img_dim, img_dim)
        self.transform = transform

    def __getitem__(self, idx):
        """
        每次迭代时返回数据和对应的标签
        """
        img_path = self.dir + self.datalist[idx]
        img = io.imread(img_path)
        img = transform.resize(img, self.image_size)
        img = img.transpose((2, 0, 1))  # 转换为通道优先
        img = img.astype("float32")

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        """
        返回整个数据集的总数
        """
        return len(self.datalist)


# 定义数据集
train_dataset = DataGenerater()

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,  # 使用 pin_memory 可以加速数据传输
    drop_last=True,
)
```
4. train.py
```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from network import *
from dataloader import *
from tqdm import tqdm
import multiprocessing


img_dim = 128 # 图像尺寸
lr = 0.0002 # 学习率
epochs = 5  # 训练轮数
batch_size = 128    # 批量大小
G_DIMENSION = 100   # 生成器输入的维度
beta1 = 0.5 # Adam优化器的第一个beta参数
beta2 = 0.999   # Adam优化器的第二个beta参数
output_path = 'output'  # 输出路径
real_label = 1  # 真实标签
fake_label = 0  # 假标签

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(): 
    print(f"当前使用设备: {device}")
    if device.type == 'cuda':
        print(f"CUDA GPU 型号: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 当前运行在 CPU 上，会非常慢！请检查 CUDA 是否安装并启用。")

    # 定义模型
    netD = Discriminator().to(device)   #  判别器
    netG = Generator().to(device)  # 生成器

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # 训练过程
    losses = [[], []]
    plt.ioff()
    now = 0 # now 变量似乎未使用，可以考虑移除
    for epoch in range(epochs):
        for batch_id, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
            ############################
            # (1) 更新判别器 D
            ###########################
            optimizerD.zero_grad()
            real_cpu = data.to(device)
            current_batch_size = real_cpu.size(0) 
            label = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1) # 确保输出是一维的
            
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(current_batch_size, G_DIMENSION, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) 更新生成器 G
            ###########################
            optimizerG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            losses[0].append(errD.item())
            losses[1].append(errG.item())

            if batch_id % 100 == 0:
                try:
                    if now % 500 == 0:
                        plt.figure(figsize=(15, 6))
                        x_ = np.arange(len(losses[0]))
                        plt.title('Generator and Discriminator Loss During Training')
                        plt.xlabel('Number of Batch')
                        plt.plot(x_, np.array(losses[0]), label='D Loss')
                        plt.plot(x_, np.array(losses[1]), label='G Loss')
                        plt.legend()
                        plt.savefig(os.path.join(output_path, 'loss_curve_temp.png'))
                        plt.close() # 关闭图像以释放内存
                    now += 1
                except IOError:
                    print(IOError)

    # 训练结束后的操作
    plt.close()
    plt.figure(figsize=(15, 6))
    x_axis = np.arange(len(losses[0]))
    plt.title('Generator and Discriminator Loss During Training')
    plt.xlabel('Number of Batch')
    plt.plot(x_axis, np.array(losses[0]), label='D Loss')
    plt.plot(x_axis, np.array(losses[1]), label='G Loss')
    plt.legend()
    plt.savefig('Generator_and_Discriminator_Loss_During_Training.png')
    plt.close()

    torch.save(netG.state_dict(), "generator.params")
    print("Generator model saved as generator.params")

if __name__ == '__main__':
    multiprocessing.freeze_support() # 添加 freeze_support()
    main() # 调用主函数
``` 
