import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from  models import Generator
from datasets import ImageDataset
from torchvision.utils import save_image
import torch.multiprocessing as mp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    netG_A2B.load_state_dict(torch.load("models\\netG_A2B.pth"))
    netG_B2A.load_state_dict(torch.load("models\\netG_B2A.pth"))

    netG_A2B.eval()
    netG_B2A.eval()

    size = 256
    input_A = torch.ones([1, 3, size, size], dtype=torch.float).to(device)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float).to(device)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    data_root = "img"
    dataloader = DataLoader(ImageDataset(data_root, transforms_, "test"),
                                batch_size=1, shuffle=False, num_workers=8)

    for i, batch in enumerate(dataloader):
        real_A = input_A.copy_(batch['A'].clone().detach()).to(device)
        real_B = input_B.copy_(batch['B'].clone().detach()).to(device)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        save_image(fake_A, "outputs\\A\\{}.png".format(i))
        save_image(fake_B, "outputs\\B\\{}.png".format(i))
        print(i)

if __name__ == '__main__':
    # 在主程序开始前，设置启动方法为 spawn
    mp.set_start_method('spawn')
    main()

