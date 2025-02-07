import os, pickle, torch, urllib, sys, types, glob
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.transforms import (
    Compose, 
    Lambda,
    Resize,
)
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
    UniformTemporalSubsample,
    ShortSideScale,
)


class Transform:
    """
    Class to define the transformation of the video input for the model and make it reusable.
    """
    def __init__(self):
        pass

    def slowfast_transform(self, num_frames, downsample_size, mean, std):
        return  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # TODO: must be first 24 frames in each second? 
                    # Do not work with uniformly separated samples? 
                    # If not, then try to update class based on source code.
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    Resize(downsample_size),
                    PackPathway()
                ]
            ),
        )

    def resnet_transform(self, mean, std):
        return  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # TODO: must be first frame in each second? 
                    # Do not work with uniformly separated samples? 
                    # If not, then try to update class based on source code.
                    UniformTemporalSubsample(1),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                ]
            ),
        )


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor, slowfast_alpha: int = 4):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class VideoDataset(Dataset):
    def __init__(self, root_dir):
        """
        Implementation of Dataset class for LIVE NETFLIX - II dataset.
        LIVE Netflix Video Quality of Experience dataset lacking.
        More info found @ http://live.ece.utexas.edu/research/LIVE_NFLX_II/live_nflx_plus.html
        and @ http://live.ece.utexas.edu/research/LIVE_NFLXStudy/nflx_index.html.
        """
        self.transforms = [
            Transform().slowfast_transform,
            Transform().resnet_transform,
        ]
        self.root_dir = root_dir
        pkl_files = glob.glob(
            os.path.join(self.root_dir, 'Dataset_Information/Pkl_Files/*.pkl')
        )
        self.annotations = self.load_annotations(pkl_files)

    def load_annotations(self, pkl_files):
        annotations = []
        for file in pkl_files:
            with open(file, 'rb') as f:
                annotations.append(pickle.load(f, encoding='latin1'))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        downsample_size = (224, 224)
        mean = [0.45, 0.45, 0.45] # TODO: check if normalization parameters are correct.
        std = [0.225, 0.225, 0.225]
        frame_sample = 32 # TODO: sample 24 first frames in each second for slowfast.

        slowfast_transform = self.transforms[0](frame_sample, downsample_size, mean, std)
        resnet_transform = self.transforms[1](mean, std) # TODO: sample first frame in each second for resnet.

        clip_duration = self.annotations[idx]['video_duration_sec']
        start_sec = 0
        # end_sec = start_sec + clip_duration
        end_sec = 2 # Used to make model run im slow PC.

        video_path = os.path.join(self.root_dir, 'assets_mp4_individual', self.annotations[idx]['distorted_mp4_video'])
        video = EncodedVideo.from_path(video_path)
        video_data = [
            video.get_clip(start_sec=start_sec, end_sec=end_sec),
            video.get_clip(start_sec=start_sec, end_sec=end_sec)
        ]

        slowfast_transform(video_data[0])
        resnet_transform(video_data[1])

        return [v['video'] for v in video_data]