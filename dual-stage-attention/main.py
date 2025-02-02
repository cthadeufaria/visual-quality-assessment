import torch
import torch.nn as nn
import urllib.request
import sys
import types
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

# Fixed module import error using the impl. @ https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-2439896362
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import rgb_to_grayscale

functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from backbone import Backbone


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = Backbone().to(device)
    backbone.eval()

    # Define input transform:
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    slowfast_alpha = 4
    num_clips = 10
    num_crops = 3

    class PackPathway(nn.Module):
        """
        Transform for converting video frames as a list of tensors. 
        """
        def __init__(self):
            super().__init__()
            
        def forward(self, frames: torch.Tensor):
            fast_pathway = frames
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                frames,
                1,
                torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
            return frame_list

    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        ),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    # Download video for testing:
    url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
    video_path = 'dual-stage-attention/videos/archery.mp4'

    try: 
        urllib.URLopener().retrieve(url_link, video_path)

    except: 
        urllib.request.urlretrieve(url_link, video_path)

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]

    output = backbone(inputs)
    print('output.shape:', output.shape)


if __name__ == "__main__":
    main()