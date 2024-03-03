import torch
from diffusers import DDPMScheduler
from datahandling.transforms import to_class_int
from util.plot_tools import show_and_save

def sample_single_image(model, img_size, device, timesteps, label, target_labels, epoch):
    with torch.no_grad():
        scheduler = DDPMScheduler(num_train_timesteps=timesteps)
        image = torch.randn(1, 3, img_size, img_size).to(device)
        label = to_class_int(label, target_labels).to(device)

        for t in scheduler.timesteps:
            output = model(image, timestep=t, class_labels=label, return_dict=False)[0]
            image = scheduler.step(output, t, image).prev_sample

        image = ((image / 2) + .5).clamp(0,1 )
        image = image.permute(0,2,3,1).squeeze().detach().cpu().numpy()
        output = (image * 255).astype(int)
        show_and_save(output, epoch, label)
        