from diffusers import UNet2DModel

def cxr_unet(img_size, label_count):
    model = UNet2DModel(
        sample_size=img_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        num_class_embeds=label_count,
        norm_num_groups=32,
        class_embed_type="timestep",
        down_block_types=(
            "DownBlock2D", 
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

def train_one_epoch(model):
    model.train()