import matplotlib.pyplot as plt
import torch



def add_noise(inputs,noise_factor=0.3):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy

def plot_result(img,rec_img):
    ### Plot progress
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(img.squeeze().numpy(), cmap='gist_gray')
    axs[0].set_title('Original image')
    axs[1].imshow(rec_img.squeeze().numpy(), cmap='gist_gray')
    plt.tight_layout()
    plt.pause(0.1)

    plt.show()
    plt.close()

