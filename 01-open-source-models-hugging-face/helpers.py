from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import matplotlib.pyplot as plt


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def show_mask(mask, ax, random_color=False):
    """Display a mask on a matplotlib axis"""
    if random_color:
        color = np.concatenate([np.random.random(3),
                                np.array([0.6])],
                               axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    # Convert to numpy array to avoid deprecation warning
    mask_np = np.array(mask)
    mask_image = mask_np.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_mask_on_image(raw_image, mask, output_path=None, return_image=False, show_all_masks=True):
    """
    Display or save a mask overlay on an image.

    Args:
        raw_image: PIL Image or numpy array
        mask: Mask tensor, array, or pipeline output dict with 'masks' key
        output_path: Path to save the image (if None, will display instead)
        return_image: Whether to return the image as PIL Image
        show_all_masks: If mask is a dict/list, whether to show all masks or just the first

    Returns:
        PIL Image if return_image=True, otherwise None
    """
    # Handle pipeline output (dict with 'masks' key)
    if isinstance(mask, dict) and 'masks' in mask:
        masks = mask['masks']
    elif isinstance(mask, list):
        masks = mask
    else:
        masks = [mask]

    fig, axes = plt.subplots(1, 1, figsize=(15, 15))
    axes.imshow(np.array(raw_image))

    # Show all masks or just the first one
    masks_to_show = masks if show_all_masks else masks[:1]

    for i, single_mask in enumerate(masks_to_show):
        if not isinstance(single_mask, torch.Tensor):
            single_mask = torch.Tensor(single_mask)

        if len(single_mask.shape) == 4:
            single_mask = single_mask.squeeze()

        single_mask = single_mask.cpu().detach()
        # Use random colors for multiple masks to distinguish them
        show_mask(single_mask, axes, random_color=(len(masks_to_show) > 1))

    axes.axis("off")

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved mask overlay to: {output_path}")
    else:
        plt.show()

    result_image = None
    if return_image:
        result_image = fig2img(fig)

    plt.close(fig)

    return result_image


def show_depth_map(raw_image, depth_map, output_path=None, return_image=False, colormap='inferno', show_original=True, show_colorbar=False):
    """
    Display or save a depth map visualization.

    Args:
        raw_image: PIL Image or numpy array - original input image
        depth_map: numpy array or torch tensor - depth predictions (H, W)
        output_path: Path to save the image (if None, will display instead)
        return_image: Whether to return the image as PIL Image
        colormap: Matplotlib colormap name (default: 'inferno')
        show_original: If True, display side-by-side (original + depth);
                      If False, display depth map only (default: True)
        show_colorbar: If True, display the colorbar scale legend (default: False)

    Returns:
        PIL Image if return_image=True, otherwise None

    Examples:
        # Side-by-side view (default)
        show_depth_map(image, depth, output_path="comparison.png")

        # Depth-only view
        show_depth_map(image, depth, output_path="depth_only.png", show_original=False)
    """
    # Convert depth_map to numpy if it's a tensor
    if torch.is_tensor(depth_map):
        depth_map = depth_map.cpu().detach().numpy()

    # Normalize depth map to 0-1 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_map)

    # Create figure based on display mode
    if show_original:
        # Side-by-side mode (current behavior)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        depth_ax = axes[1]

        # Show original image
        axes[0].imshow(np.array(raw_image))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
    else:
        # Depth-only mode (new behavior)
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        depth_ax = axes

    # Show depth map with colormap (works for both modes)
    im = depth_ax.imshow(depth_normalized, cmap=colormap)
    depth_ax.axis("off")

    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im, ax=depth_ax, fraction=0.046, pad=0.04)

    # Save or display
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved depth map to: {output_path}")
    else:
        plt.show()

    # Return image if requested
    result_image = None
    if return_image:
        result_image = fig2img(fig)

    plt.close(fig)
    return result_image