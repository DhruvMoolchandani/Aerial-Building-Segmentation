import torch
import segmentation_models_pytorch as smp


def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,  # channel 0 = mask, channel 1 = boundary
        activation=None,
    )


def build_model_old_style():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def build_model_from_checkpoint(old_checkpoint_path, device="cuda"):
    """
    Loads encoder + decoder weights from your previous 1-channel model into the
    new 2-channel model, skipping only the final segmentation_head layer
    (which has a shape mismatch: old=1 channel, new=2 channels).
    """
    model = build_model().to(device)
    old_state = torch.load(old_checkpoint_path, map_location=device)
    new_state = model.state_dict()

    compatible_state = {
        k: v for k, v in old_state.items()
        if k in new_state and v.shape == new_state[k].shape
    }
    skipped = [k for k in old_state if k not in compatible_state]

    new_state.update(compatible_state)
    model.load_state_dict(new_state)

    print(f"Loaded {len(compatible_state)} matching layers, "
          f"skipped {len(skipped)} mismatched layers (expected: segmentation_head)")
    print(f"Skipped keys: {skipped}")
    return model

"""Without TTA: 0.7243
With TTA:    0.7349
Difference:  +0.0106-results"""