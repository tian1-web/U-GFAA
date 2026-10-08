import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from lib.utils.freq_aug import amp_spectrum_mix

# ======================= [ 配置区域 ] =======================

IMG_SRC_PATH = "/root/shared-nvme/datasets/cityscapes/leftImg8bit/train/erfurt/erfurt_000042_000019_leftImg8bit.png"
LABEL_SRC_PATH = "/root/shared-nvme/datasets/cityscapes/gtFine/train/erfurt/erfurt_000042_000019_gtFine_labelTrainIds.png"
IMG_GEN_PATH = "/root/shared-nvme/datasets/DTWP_ADE_final/leftImg8bit/train/erfurt/erfurt_000042_000019_leftImg8bit.png"

OUTPUT_PATH = "augmentation_comparison.png"

# 轮廓配置
CONTOUR_COLOR = (0, 255, 0)   # 绿色
CONTOUR_THICKNESS = 2

# 放大框配置: (x, y, w, h)
ZOOM_BOX_1 = (1100, 350, 200, 150)
ZOOM_BOX_2 = (400, 600, 200, 150)

# 是否额外输出第二张局部对比图
SAVE_SECOND_ZOOM = True

# U-SAFA 幅度混合强度
ALPHA = 0.6

# ========================================================


def check_paths():
    paths = {
        "IMG_SRC_PATH": IMG_SRC_PATH,
        "LABEL_SRC_PATH": LABEL_SRC_PATH,
        "IMG_GEN_PATH": IMG_GEN_PATH,
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")


def generate_usafa_image(img_src_pil, img_gen_pil, alpha=0.6):
    """生成 U-SAFA 增强图像"""
    print("Generating U-SAFA image...")

    img_t = torch.from_numpy(np.array(img_src_pil)).permute(2, 0, 1).float()
    gen_t = torch.from_numpy(np.array(img_gen_pil)).permute(2, 0, 1).float()

    aug_t = amp_spectrum_mix(img_t, gen_t, alpha=alpha)
    aug_t = torch.clamp(aug_t, 0, 255)

    img_usafa_pil = Image.fromarray(
        aug_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    )

    print("U-SAFA image generated.")
    return img_usafa_pil


def draw_contours(image_pil, label_path, color=(0, 255, 0), thickness=2):
    """在 RGB 图像上叠加 GT 轮廓"""
    image_np = np.array(image_pil).copy()
    label = np.array(Image.open(label_path))

    unique_labels = np.unique(label)

    for label_id in unique_labels:
        # 255 = ignore；你如果也想忽略 road(0)，就保留下面这一行
        if label_id in [0, 255]:
            continue

        mask = (label == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_np, contours, -1, color, thickness)

    return Image.fromarray(image_np)


def add_zoom_box(ax, box, color):
    x, y, w, h = box
    rect = plt.Rectangle((x, y), w, h, edgecolor=color, facecolor='none', linewidth=3)
    ax.add_patch(rect)


def crop_box(img_pil, box):
    x, y, w, h = box
    return img_pil.crop((x, y, x + w, y + h))


def style_ax(ax, title=None, spine_color=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=18, pad=12)
    if spine_color is not None:
        for k in ax.spines:
            ax.spines[k].set_visible(True)
            ax.spines[k].set_linewidth(3)
            ax.spines[k].set_color(spine_color)
    else:
        for k in ax.spines:
            ax.spines[k].set_visible(False)


def save_main_figure(img_src_contour, img_gen_contour, img_usafa_contour, output_path):
    """保存主对比图：上排整体，下排局部放大 1"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    # 上排：整体图
    axes[0, 0].imshow(img_src_contour)
    axes[0, 1].imshow(img_gen_contour)
    axes[0, 2].imshow(img_usafa_contour)

    style_ax(axes[0, 0], "Source Image")
    style_ax(axes[0, 1], "Conventional Style Perturbation")
    style_ax(axes[0, 2], "Our Structure-Aligned Augmentation")

    add_zoom_box(axes[0, 0], ZOOM_BOX_1, "lime")
    add_zoom_box(axes[0, 1], ZOOM_BOX_1, "lime")
    add_zoom_box(axes[0, 2], ZOOM_BOX_1, "lime")

    # 下排：局部放大 1
    axes[1, 0].imshow(crop_box(img_src_contour, ZOOM_BOX_1))
    axes[1, 1].imshow(crop_box(img_gen_contour, ZOOM_BOX_1))
    axes[1, 2].imshow(crop_box(img_usafa_contour, ZOOM_BOX_1))

    style_ax(axes[1, 0], "Zoomed Boundary Region", "lime")
    style_ax(axes[1, 1], "Potential Misalignment", "lime")
    style_ax(axes[1, 2], "Structure Preserved", "lime")

    fig.suptitle(
        "Comparison Between Conventional Style Perturbation and Our Structure-Aligned Frequency Augmentation",
        fontsize=20,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Main figure saved to: {output_path}")


def save_second_zoom_figure(img_src_contour, img_gen_contour, img_usafa_contour, output_path):
    """可选：保存第二个局部区域对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    axes[0].imshow(crop_box(img_src_contour, ZOOM_BOX_2))
    axes[1].imshow(crop_box(img_gen_contour, ZOOM_BOX_2))
    axes[2].imshow(crop_box(img_usafa_contour, ZOOM_BOX_2))

    style_ax(axes[0], "Source Local Detail", "cyan")
    style_ax(axes[1], "Conventional Perturbation", "cyan")
    style_ax(axes[2], "Our U-SAFA", "cyan")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Second zoom figure saved to: {output_path}")


def create_comparison_figure():
    check_paths()

    img_src = Image.open(IMG_SRC_PATH).convert("RGB")
    img_gen = Image.open(IMG_GEN_PATH).convert("RGB")

    img_usafa = generate_usafa_image(img_src, img_gen, alpha=ALPHA)

    print("Drawing contours...")
    img_src_contour = draw_contours(
        img_src, LABEL_SRC_PATH, CONTOUR_COLOR, CONTOUR_THICKNESS
    )
    img_gen_contour = draw_contours(
        img_gen, LABEL_SRC_PATH, CONTOUR_COLOR, CONTOUR_THICKNESS
    )
    img_usafa_contour = draw_contours(
        img_usafa, LABEL_SRC_PATH, CONTOUR_COLOR, CONTOUR_THICKNESS
    )

    save_main_figure(
        img_src_contour,
        img_gen_contour,
        img_usafa_contour,
        OUTPUT_PATH
    )

    if SAVE_SECOND_ZOOM:
        second_output = os.path.splitext(OUTPUT_PATH)[0] + "_zoom2.png"
        save_second_zoom_figure(
            img_src_contour,
            img_gen_contour,
            img_usafa_contour,
            second_output
        )


if __name__ == "__main__":
    create_comparison_figure()