import math

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError
from torchvision import transforms
from torchvision.transforms import Resize

DEFAULT_METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]


def mask_decode(encoded_mask: list[int], image_shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for index in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[index + 1], length - encoded_mask[index])
        for offset in range(splice_len):
            mask_array[encoded_mask[index] + offset] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def build_mask(mask_rle: list[int]) -> np.ndarray:
    mask = mask_decode(mask_rle)
    return mask[:, :, np.newaxis].repeat(3, axis=2)


def crop_for_evaluation(image):
    if image.size[0] != image.size[1]:
        return image.crop((image.size[0] - 512, image.size[1] - 512, image.size[0], image.size[1]))
    return image


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected a scalar metric value, got shape {value.shape}")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


class VitExtractor:
    BLOCK_KEY = "block"
    ATTN_KEY = "attn"
    PATCH_IMD_KEY = "patch_imd"
    QKV_KEY = "qkv"
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()
        self.device = device

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input_tensor, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, input_tensor, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, input_tensor, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_attn_output(model, input_tensor, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        _, _, _, width = input_img_shape
        return width // self.get_patch_size()

    def get_height_patch_num(self, input_img_shape):
        _, _, height, _ = input_img_shape
        return height // self.get_patch_size()

    def get_patch_num(self, input_img_shape):
        return 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        return qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        return self.get_keys_from_qkv(qkv_features, input_img.shape)

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        heads, tokens, dim = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(tokens, heads * dim)
        return self.attn_cosine_sim(concatenated_keys[None, None, ...])

    def attn_cosine_sim(self, x, eps=1e-08):
        x = x[0]
        norm1 = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
        return (x @ x.permute(0, 2, 1)) / factor


class LossG(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.extractor = VitExtractor(model_name=cfg["dino_model_name"], device=device)

        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg["dino_global_patch_size"], max_size=480)
        self.global_transform = transforms.Compose([global_resize_transform, imagenet_norm])

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a_tensor, b_tensor in zip(inputs, outputs):
            a_tensor = self.global_transform(a_tensor)
            b_tensor = self.global_transform(b_tensor)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a_tensor.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b_tensor.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss


class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.structure_distance_metric_calculator = LossG(
            cfg={
                "dino_model_name": "dino_vitb8",
                "dino_global_patch_size": 224,
            },
            device=device,
        )

    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)
        if mask is not None:
            img = np.uint8(img * np.array(mask))
        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        return self.clip_metric_calculator(img_tensor, txt).cpu().item()

    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor).cpu().item()

    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1).cpu().item()

    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)
        return self.mse_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous()).cpu().item()

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor).cpu().item()

    def calculate_structure_distance(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)
        img_gt = np.array(img_gt).astype(np.float32)
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.from_numpy(np.transpose(img_pred, axes=(2, 0, 1))).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.from_numpy(np.transpose(img_gt, axes=(2, 0, 1))).unsqueeze(0).to(self.device)
        structure_distance = self.structure_distance_metric_calculator.calculate_global_ssim_loss(
            img_gt_tensor,
            img_pred_tensor,
        )
        return structure_distance.data.cpu().numpy()


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_psnr(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_lpips(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_mse(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_ssim(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "structure_distance_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1 - src_mask, 1 - tgt_mask)
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return float("nan")
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    raise ValueError(f"Unsupported metric: {metric}")


def compute_metrics(metrics_calculator, metrics, src_image, tgt_image, mask, src_prompt, tgt_prompt):
    results = {}
    for metric_name in metrics:
        metric_value = calculate_metric(
            metrics_calculator,
            metric_name,
            src_image,
            tgt_image,
            mask,
            mask,
            src_prompt,
            tgt_prompt,
        )
        results[metric_name] = safe_float(metric_value)
    return results
