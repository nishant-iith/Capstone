import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_results(unstained_path, virtual_path, real_path, output_path):
    u = Image.open(unstained_path)
    v = Image.open(virtual_path)
    r = Image.open(real_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(u)
    axes[0].set_title("Unstained")
    axes[1].imshow(v)
    axes[1].set_title("Virtual Stain")
    axes[2].imshow(r)
    axes[2].set_title("Real Stain")
    
    for ax in axes:
        ax.axis('off')
        
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    print("Visualization script ready.")
