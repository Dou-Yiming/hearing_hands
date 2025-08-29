import os
import os.path as osp
from omegaconf import OmegaConf
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from ipdb import set_trace as st
import sys
import diff_foley
from diff_foley.util import instantiate_from_config
from .cavp_modules import *

from ..diffusionmodules.attention_openai import SpatialTransformer_Cond


class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_size, context_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


class CAVP_Video_Encoder(nn.Module):
    def __init__(
        self,
        video_encode="Slowonly_pool",
        spec_encode="cnn14_pool",
        embed_dim: int = 32,
        video_pretrained: bool = True,
        audio_pretrained: bool = True,
    ):
        super().__init__()

        self.video_encode = video_encode
        self.spec_encode = spec_encode

        # 1). Video Encoder:
        assert self.video_encode == "Slowonly_pool"
        self.video_encoder = ResNet3dSlowOnly(
            depth=18, pretrained=True
        )  # Doesn't matter to set pretrained=None, since we will load CAVP weight outside.

        # Video Project & Pooling Head:
        self.video_project_head = nn.Linear(512, embed_dim)
        self.video_pool = nn.MaxPool1d(kernel_size=16)

    def encode_video(self, video, normalize=True, pool=False):
        # Video: B x T x 3 x H x W
        assert self.video_encode == "Slowonly_pool"
        video = video.permute(0, 2, 1, 3, 4)
        video_feat = self.video_encoder(video)
        bs, c, t, _, _ = video_feat.shape
        video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)  # B T C
        video_feat = self.video_project_head(video_feat)

        # Pooling:
        if pool:
            video_feat = self.video_pool(video_feat.permute(0, 2, 1)).squeeze(2)

        # Normalize:
        if normalize:
            video_feat = F.normalize(video_feat, dim=-1)

        return video_feat


class Video_Feat_Encoder(nn.Module):
    """Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, latent_len=3760):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(origin_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        # Revise the shape here:
        x = self.embedder(x)  # B x 117 x C
        # x = torch.randn(x.shape[0], 3760, 128).to(x.device)

        return x


class Video_Feat_Encoder_simple(nn.Module):
    """Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, latent_len=3760):
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))

    def forward(self, x):
        # Revise the shape here:
        x = self.embedder(x)  # B x 117 x C
        # x = torch.randn(x.shape[0], 3760, 128).to(x.device)
        return x


class Video_Feat_Encoder_Posembed(nn.Module):
    """Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        from ipdb import set_trace as st

        st()
        # Revise the shape here:
        bs, seq_len, c = x.shape
        x = self.embedder(x)  # B x 117 x C
        pos_embedding = self.pos_emb(
            torch.arange(seq_len, device=x.device).reshape(1, -1)
        ).repeat(bs, 1, 1)
        x = x + pos_embedding
        return x


class Video_Feat_Encoder_NoPosembed(nn.Module):
    """Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(origin_dim, embed_dim))

    def forward(self, x):
        # Revise the shape here:
        x = self.embedder(x)  # B x 117 x C

        return x


class Video_Feat_Encoder_TimeUpsample_NoPosembed_HandPose(nn.Module):
    def __init__(
        self,
        cavp_feat_origin_dim,
        clip_feat_origin_dim,
        clip_local_feat_origin_dim,
        hand_pose_origin_dim,
        dino_feat_origin_dim,
        dino_local_feat_origin_dim,
        siglip_feat_origin_dim,
        siglip_local_feat_origin_dim,
        embed_dim,
        use_cavp_feat,
        use_hand_pose,
        use_clip_feat,
        use_clip_local_feat,
        use_dino_feat,
        use_dino_local_feat,
        use_siglip_feat,
        use_siglip_local_feat,
    ):
        super().__init__()

        self.use_cavp_feat = use_cavp_feat
        self.use_hand_pose = use_hand_pose
        self.use_clip_feat = use_clip_feat
        self.use_clip_local_feat = use_clip_local_feat
        self.use_dino_feat = use_dino_feat
        self.use_dino_local_feat = use_dino_local_feat
        self.use_siglip_feat = use_siglip_feat
        self.use_siglip_local_feat = use_siglip_local_feat

        input_feat_dim = 0
        if self.use_cavp_feat:
            input_feat_dim += cavp_feat_origin_dim
        if self.use_clip_feat:
            input_feat_dim += clip_feat_origin_dim
        if self.use_clip_local_feat:
            input_feat_dim += clip_local_feat_origin_dim
        if self.use_siglip_feat:
            input_feat_dim += siglip_feat_origin_dim
        if self.use_siglip_local_feat:
            input_feat_dim += siglip_local_feat_origin_dim
        if self.use_dino_feat:
            input_feat_dim += dino_feat_origin_dim
        if self.use_dino_local_feat:
            input_feat_dim += dino_local_feat_origin_dim
        self.use_video_feat = input_feat_dim > 0
        if self.use_video_feat:
            self.video_feat_embedder = nn.Sequential(
                nn.Linear(input_feat_dim, embed_dim),
            )
        self.hand_pose_embedder = nn.Sequential(
            nn.Linear(hand_pose_origin_dim, embed_dim),
        )

        print(
            f"Use cavp feature: {self.use_cavp_feat}, Use hand pose: {self.use_hand_pose}, Use clip feature: {self.use_clip_feat}, Use clip local feature: {self.use_clip_local_feat}, Use siglip feature: {self.use_siglip_feat}, Use siglip local feature: {self.use_siglip_local_feat}, Use dino feature: {self.use_dino_feat}, Use dino local feature: {self.use_dino_local_feat}"
        )
        print(f"Input feat dim: {input_feat_dim}")
        self.up_sample = nn.Upsample(size=250, mode="nearest")

    def forward(self, x):
        mix_video_feat = x["mix_video_feat"].cuda()

        mix_hand_pose = (
            x["mix_hand_pose"]
            if self.use_hand_pose
            else torch.zeros_like(x["mix_hand_pose"]).to(x["mix_hand_pose"].device)
        ).cuda()
        
        hand_pose_embed = self.hand_pose_embedder(mix_hand_pose)
        hand_pose_embed = hand_pose_embed / torch.norm(
            hand_pose_embed, dim=-1, keepdim=True
        )
        if self.use_video_feat:
            video_feat_embed = self.video_feat_embedder(mix_video_feat)
            video_feat_embed = rearrange(video_feat_embed, "b t c -> b c t")
            video_feat_embed = self.up_sample(video_feat_embed)
            video_feat_embed = rearrange(video_feat_embed, "b c t -> b t c")

        hand_pose_embed = rearrange(hand_pose_embed, "b t c -> b c t")
        hand_pose_embed = self.up_sample(hand_pose_embed)
        hand_pose_embed = rearrange(hand_pose_embed, "b c t -> b t c")

        if self.use_video_feat:
            feat_embed = video_feat_embed + hand_pose_embed
        else:
            feat_embed = hand_pose_embed

        return feat_embed


class Video_Feat_Encoder_TimeUpsample_NoPosembed_HandPose_TrainableVideoEncoder(
    nn.Module
):
    def __init__(
        self,
        video_feat_origin_dim,
        hand_pose_origin_dim,
        embed_dim,
        video_encoder_config_path,
    ):
        super().__init__()
        self.video_feat_embedder = nn.Sequential(
            nn.Linear(video_feat_origin_dim, embed_dim),
        )
        self.hand_pose_embedder = nn.Sequential(
            nn.Linear(hand_pose_origin_dim, embed_dim)
        )
        # load video encoder
        # self.video_encoder = CAVP_Video_Encoder()
        # ckpt_path = "/home/ymdou/SaRF/data_capture/cavp/pretrained_cavp.ckpt"
        # ckpt = torch.load(ckpt_path, map_location="cpu")
        # state_dict = ckpt["state_dict"]
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # missing, unexpected = self.video_encoder.load_state_dict(
        #     state_dict, strict=False
        # )
        # print(
        #     f"Restored video encoder from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        # )

        self.up_sample = nn.Upsample(size=250, mode="nearest")

    def forward(self, x):
        # try:
        # except:
        #     st()
        #     print('')
        # video_feat = self.video_encoder.encode_video(x["mix_video_frames"])
        # video_feat = torch.concat([x["mix_video_feat"], video_feat], dim=-1)
        video_feat = x["mix_video_feat"]
        video_feat_embed = self.video_feat_embedder(video_feat)

        hand_pose_embed = self.hand_pose_embedder(x["mix_hand_pose"])
        hand_pose_embed = F.normalize(hand_pose_embed, dim=-1)

        video_feat_embed = rearrange(video_feat_embed, "b t c -> b c t")
        video_feat_embed = self.up_sample(video_feat_embed)
        video_feat_embed = rearrange(video_feat_embed, "b c t -> b t c")

        hand_pose_embed = rearrange(hand_pose_embed, "b t c -> b c t")
        hand_pose_embed = self.up_sample(hand_pose_embed)
        hand_pose_embed = rearrange(hand_pose_embed, "b c t -> b t c")

        video_feat_embed += hand_pose_embed

        return video_feat_embed


class Video_Feat_Encoder_TimeUpsample(nn.Module):
    """Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__()
        self.embedder = nn.Upsample(scale_factor=8, mode="nearest")

    def forward(self, x):
        # Revise the shape here:
        x = rearrange(x, "b t c -> b c t")
        x = self.embedder(x)
        x = rearrange(x, "b c t -> b t c")

        return x


class Video_Feat_Encoder_TimeUpsample_Embed(nn.Module):
    """Transform the video feat encoder"""

    def __init__(self, origin_dim, embed_dim, seq_len=215):
        super().__init__()
        self.embedder = nn.Upsample(scale_factor=8, mode="nearest")
        # self.mapper = nn.Linear(origin_dim, embed_dim)
        self.mapper = ConditionEmbedder(embed_dim, origin_dim)

    def forward(self, x):
        # Revise the shape here:
        x = rearrange(x, "b t c -> b c t")
        x = self.embedder(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.mapper(x)

        return x


class FusionNet(nn.Module):
    def __init__(self, hidden_dim, embed_dim, depth, heads=8, d_head=64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.fusion_module = SpatialTransformer_Cond(
            in_channels=hidden_dim, n_heads=heads, d_head=d_head, depth=depth
        )
        self.proj_out = nn.Sequential(nn.Linear(hidden_dim, embed_dim))

    def forward(self, video_feat, spec_feat):
        """
        Input:
            video_feat: B x L x C
            spec_feat: B x C x H x W
        Output:
            B x L x C
        """
        bs, c, h, w = spec_feat.shape
        spec_feat = spec_feat.permute(0, 2, 3, 1).reshape(bs, -1, c)
        fusion_features = self.fusion_module(video_feat, spec_feat)
        fusion_features = self.proj_out(fusion_features)
        return fusion_features


class Video_Feat_Encoder_Posembed_AR(nn.Module):
    """Transform the video feat encoder"""

    """
        Input:
            Data Dict: 
                video_feat:  B x L x C
                spec_prev_z: B x C x H x W
    """

    def __init__(self, origin_dim, hidden_dim, embed_dim, depth=2, seq_len=215):
        super().__init__()
        self.embed_video_feat = nn.Sequential(nn.Linear(origin_dim, hidden_dim))
        self.embed_spec_feat = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=hidden_dim, kernel_size=1)
        )
        self.fusion_net = FusionNet(hidden_dim, embed_dim, depth)
        self.pos_emb_video = nn.Embedding(seq_len, hidden_dim)
        self.pos_emb_spec = nn.Embedding(seq_len, hidden_dim)

    def forward(self, x):
        video_feat = x["video_feat"]
        spec_prev_z = x["spec_prev_z"]

        bs, seq_len, c = video_feat.shape
        bs, _, spec_h, spec_w = spec_prev_z.shape

        video_feat = self.embed_video_feat(video_feat)  # B x L x C
        spec_feat = self.embed_spec_feat(spec_prev_z)  # B x C' x H x W

        # Add Pos Embedding:
        pos_embed_video = self.pos_emb_video(
            torch.arange(seq_len, device=video_feat.device).reshape(1, -1)
        ).repeat(bs, 1, 1)
        video_feat = video_feat + pos_embed_video

        pos_embed_spec = (
            self.pos_emb_spec(
                torch.arange(spec_w, device=video_feat.device).reshape(1, -1)
            )
            .permute(0, 2, 1)
            .unsqueeze(2)
        )  # 1 x C x W
        pos_embed_spec = pos_embed_spec.repeat(bs, 1, spec_h, 1)
        spec_feat = spec_feat + pos_embed_spec

        # Features Fusion:  Cross Attention
        fuse_features = self.fusion_net(video_feat, spec_feat)
        return fuse_features
