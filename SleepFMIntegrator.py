from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Union
import numpy as np
from scipy.signal import resample
from sleepfm.models.models import SetTransformer
import torch
from torch import nn

@dataclass
class SleepFMConfig:
    """
    Configuration for integrating SleepFM into the George B. Moody
    PhysioNet Challenge pipeline.

    This class intentionally extends beyond the original SleepFM
    configuration so that the same object can be reused for

        Strategy 1 : Frozen feature extractions
        Strategy 2 : End-to-end fine-tuning
        Strategy 3 : Hybrid handcrafted + SleepFM model
    """

    # General

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    model_name: str = "SleepFM"
    model_version: str = "clinical_v1"
    use_amp: bool = True

    # Checkpoints

    checkpoint_path: Optional[Path] = None
    strict_checkpoint_loading: bool = False
    save_embedding_cache: bool = True
    embedding_cache_dir: Optional[Path] = None

    # Input signals

    expected_sampling_frequency: int = 128
    window_length_seconds: int = 5
    window_stride_seconds: int = 5
    normalize_signals: bool = True
    normalization_method: str = "zscore"
    clip_value: Optional[float] = 8.0

    # Patch tokenizer

    patch_size: int = 640
    embedding_dimension: int = 128

    # Transformer

    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    pooling_heads: int = 4
    dropout: float = 0.10
    maximum_sequence_length: int = 2048
    challenge_mode: bool = True
    
    # Channel handling
    
    modality_types: Sequence[str] = (
        "BAS",
        "RESP",
        "EKG",
        "EMG",
    )

    maximum_channels_per_modality: Dict[str, int] = field(
        default_factory=lambda: {
            "BAS": 8,
            "RESP": 6,
            "EKG": 2,
            "EMG": 4,
        }
    )

    channel_groups: Dict[str, List[str]] = field(default_factory=dict)
    allow_missing_modalities: bool = True
    minimum_modalities_required: int = 1
    random_channel_selection: bool = False

    # Padding

    pad_missing_channels: bool = True
    padding_value: float = 0.0
    create_padding_masks: bool = True

    # Embeddings

    return_sequence_embeddings: bool = False
    return_patient_embedding: bool = True
    embedding_dtype: torch.dtype = torch.float32

    # Window aggregation
    aggregation_method: Literal[
        "mean",
        "max",
        "attention",
        "cls",
    ] = "mean"

    # Fine-tuning

    freeze_backbone: bool = True
    train_tokenizer: bool = False
    train_spatial_transformer: bool = False
    train_temporal_transformer: bool = False
    train_prediction_head: bool = True
    unfreeze_last_n_layers: Optional[int] = None

    # Optimisation

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    gradient_clip_norm: Optional[float] = 1.0

    # Feature extractor mode

    use_cached_embeddings: bool = True
    cache_inference_embeddings: bool = True
    cache_training_embeddings: bool = False

    # Hybrid model

    handcrafted_feature_dimension: Optional[int] = None
    concatenate_handcrafted_features: bool = False
    handcrafted_projection_dimension: int = 128

    # Debugging

    verbose: bool = False
    profile_runtime: bool = False
    save_attention_maps: bool = False
    save_intermediate_embeddings: bool = False

class SleepFMPreprocessor:
    """
    Converts the Challenge signal representation into tensors expected by
    the SleepFM encoder.

    Responsibilities
    ----------------
    * validate signals
    * resample signals
    * normalize
    * split into fixed-length windows
    * pad channels
    * create padding masks
    * return tensors
    """

    def __init__(self, config: SleepFMConfig):
        self.config = config
        self.fs = config.expected_sampling_frequency
        self.window_samples = (
            config.window_length_seconds
            * config.expected_sampling_frequency
        )
        self.stride_samples = (
            config.window_stride_seconds
            * config.expected_sampling_frequency
        )

    # Public API

     def preprocess(self,modality_signals: Dict[str, Dict[str, np.ndarray]],
                     sampling_frequencies: Optional[Union[int, Dict[str, Union[int, Dict[str, int]]]]] = None, 
                     ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    #def preprocess(self,modality_signals: Dict[str, Dict[str, np.ndarray]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Parameters
        ----------
        modality_signals
            {
                "BAS":
                    {"C3": np.ndarray,"C4": ...},
                "RESP":
                    {"Airflow": ...}
            }
        Returns
        -------
        tensors
            List of tensors
                tensor.shape
                    (channels, time)
        masks
            One mask per modality.
        """

        modality_tensors = []
        modality_masks = []
        for modality in self.config.modality_types:
            signals = modality_signals.get(modality, {})
            if isinstance(sampling_frequencies, dict):
                mod_fs = sampling_frequencies.get(modality, None)
            elif isinstance(sampling_frequencies, (int, float)):
                mod_fs = sampling_frequencies
            tensor, mask = self._process_modality(modality,signals, mod_fs)

            modality_tensors.append(tensor)
            modality_masks.append(mask)

        return modality_tensors, modality_masks

    # Processing pipeline

    def _process_modality(self,modality: str,signals: Dict[str, np.ndarray],original_fs: Optional[Union[int, Dict[str, int]]] = None):

        processed = []
        for channel_name, signal in signals.items():
            chan_fs = original_fs.get(channel_name) if isinstance(original_fs, dict) else original_fs 
            signal = self._validate_signal(signal)
            signal = self._resample(signal,signal = self._resample(signal, original_fs=chan_fs))
            signal = self._normalize(signal)
            windowed = self.create_windows(signal) 
            processed.append(windowed)
            

        tensor, mask = self._stack_channels(processed,modality)

        return tensor, mask

    # Validation

    def _validate_signal(self,signal: np.ndarray) -> np.ndarray:

        signal = np.asarray(signal).astype(np.float32)
        signal = np.nan_to_num(signal)
        if signal.ndim != 1:
            raise ValueError(
                "SleepFM only accepts 1D signals."
            )
        return signal

    # Resampling

    def _resample(self,signal: np.ndarray,original_fs: Optional[int] = None) -> np.ndarray:
        if original_fs is None or original_fs == self.fs:
            return signal
        target_length = int(len(signal)* self.fs/ original_fs)
        return resample(signal, target_length)

    # Normalisation

    def _normalize(self,signal: np.ndarray) -> np.ndarray:

        if not self.config.normalize_signals:
            return signal

        if self.config.normalization_method == "zscore":
            std = signal.std()
            if std > 0:
                signal = (signal - signal.mean()) / std
                
        elif self.config.normalization_method == "minmax":
            mn = signal.min()
            mx = signal.max()
            if mx > mn:
                signal = (signal - mn) / (mx - mn)

        if self.config.clip_value is not None:
            signal = np.clip(signal,-self.config.clip_value,self.config.clip_value)

        return signal.astype(np.float32)

    # Windowing

    def create_windows(self,signal: np.ndarray) -> np.ndarray:

        windows = []
        for start in range(0,len(signal)- self.window_samples + 1,self.stride_samples):
            windows.append(signal[start:start + self.window_samples])
        if len(windows) == 0:
            return np.empty((0, self.window_samples), dtype=np.float32) 
        return np.asarray(windows, dtype=np.float32)

    # Channel padding

    def _stack_channels(self, windows_list: List[np.ndarray], modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_channels = self.config.maximum_channels_per_modality[modality]
        num_windows = windows_list[0].shape[0] if len(windows_list) > 0 and windows_list[0].size > 0 else 0
        if len(windows_list) == 0 or num_windows == 0:
            tensor = torch.zeros((max_channels, num_windows, self.window_samples), dtype=torch.float32)
            mask = torch.ones(max_channels, dtype=torch.bool)
            return tensor, mask

        stacked = np.stack(windows_list, axis=0) 
        channels = stacked.shape[0]
        padded = np.zeros((max_channels,num_windows,self.window_samples),dtype=np.float32)

        padded[:channels] = stacked

        mask = np.ones(max_channels,dtype=bool)

        mask[:channels] = False

        return (torch.from_numpy(padded),torch.from_numpy(mask))
        
class SleepFMCheckpoint:
    """
    Handles loading, saving and configuring pretrained SleepFM models.

    This class deliberately contains no preprocessing or inference logic.
    """

    def __init__(self, config: SleepFMConfig):
        self.config = config
        self.device = torch.device(config.device)

    # Checkpoint loading

    def load_backbone(self,model: nn.Module,checkpoint_path: Optional[Path] = None,) -> nn.Module:

        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint path specified."
            )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path,map_location=self.device,)

        state_dict = self._extract_state_dict(checkpoint)

        incompatible = model.load_state_dict(state_dict,strict=self.config.strict_checkpoint_loading,)

        if self.config.verbose:
            self._print_loading_report(checkpoint_path,incompatible,)

        model.to(self.device)

        return model

    # Saving

    def save_checkpoint(self,model: nn.Module,output_path: Path,metadata: Optional[Dict] = None,):

        output = {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        }

        torch.save(output,output_path,)

    # Extract state_dict

    def _extract_state_dict(self,checkpoint: Dict,) -> Dict:

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        return checkpoint

    # Freeze

    def freeze_model(self,model: nn.Module,):

        for parameter in model.parameters():
            parameter.requires_grad = False

    # Unfreeze everything

    def unfreeze_model(self,model: nn.Module,):

        for parameter in model.parameters():
            parameter.requires_grad = True

    # Unfreeze last transformer layers

    def unfreeze_last_layers(self,transformer: nn.Module,n_layers: int,):

        if n_layers <= 0:
            return

        layers = list(transformer.transformer_encoder.layers)

        for layer in layers[-n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    # Freeze tokenizer only

    def freeze_tokenizer(self,model: nn.Module,):

        if not hasattr(model, "patch_embedding"):
            return

        for parameter in model.patch_embedding.parameters():
            parameter.requires_grad = False

    # Unfreeze tokenizer

    def unfreeze_tokenizer(self,model: nn.Module,):

        if not hasattr(model, "patch_embedding"):
            return

        for parameter in model.patch_embedding.parameters():
            parameter.requires_grad = True

    # Parameter summary

    def trainable_parameter_count(self,model: nn.Module,) -> int:

        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    # Report

    def _print_loading_report(self,checkpoint_path: Path,incompatible,):

        print()
        print("=" * 60)
        print("SleepFM checkpoint loaded")
        print(checkpoint_path)
        print()
        print(
            "Missing keys:",
            len(incompatible.missing_keys),
        )
        print(
            "Unexpected keys:",
            len(incompatible.unexpected_keys),
        )
        print("=" * 60)
        
class SleepFMEncoder(nn.Module):
    """
    Wrapper around the pretrained SleepFM SetTransformer.

    Responsibilities
    ----------------
    * Construct the original SleepFM backbone.
    * Load pretrained checkpoints.
    * Freeze / unfreeze parameters.
    * Provide a clean interface for downstream strategies.

    This class intentionally does NOT perform preprocessing.
    That is delegated to SleepFMPreprocessor.
    """

    def __init__(self,config: SleepFMConfig,checkpoint_handler: Optional[SleepFMCheckpoint] = None,) -> None:

        super().__init__()

        self.config = config
        self.checkpoint_handler = checkpoint_handler

        
        # Build original SleepFM backbone.
        
        self.backbone = self._build_backbone()

        
        # Load checkpoint if requested.
        
        if (checkpoint_handler is not None and config.load_pretrained):
            checkpoint_handler.load_backbone(self.backbone)

        # Freeze if requested.
        if config.freeze_backbone:
            self.freeze_backbone()

    # Backbone construction
    
    def _build_backbone(self) -> SetTransformer:
        """
        Construct the original SleepFM SetTransformer.

        This mirrors the architecture used during SleepFM pretraining.
        """

        backbone = SetTransformer(
            in_channels=self.config.max_channels,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            pooling_head=self.config.pooling_heads,
            dropout=self.config.dropout,
            max_seq_length=self.config.max_sequence_length,
        )

        return backbone

    # Checkpoint handling
 
    def load_checkpoint(self,checkpoint: str | Path,strict: bool = False) -> None:
        """
        Load a pretrained checkpoint into the backbone.
        """

        if self.checkpoint_handler is None:
            raise RuntimeError("SleepFMCheckpoint has not been supplied.")

        self.checkpoint_handler.load_backbone(
            self.backbone,
            checkpoint_path=checkpoint,
            strict=strict,
        )

    # Freezing

    def freeze_backbone(self) -> None:
        """
        Freeze every parameter in the SleepFM backbone.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the complete backbone.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def freeze_tokenizer(self) -> None:
        """
        Freeze only the CNN tokenizer.
        """

        for parameter in self.backbone.patch_embedding.parameters():
            parameter.requires_grad = False

    def freeze_transformer(self) -> None:
        """
        Freeze only the transformer encoder.
        """

        modules = [
            self.backbone.spatial_pooling,
            self.backbone.transformer_encoder,
            self.backbone.temporal_pooling,
        ]

        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def unfreeze_last_transformer_layer(self) -> None:
        """
        Fine-tuning strategy:
        keep everything frozen except the final transformer block.
        """

        self.freeze_backbone()

        last_layer = self.backbone.transformer_encoder.layers[-1]

        for parameter in last_layer.parameters():
            parameter.requires_grad = True

    # Utilities

    @property
    def device(self) -> torch.device:
        """
        Return the device on which the backbone resides.
        """

        return next(self.backbone.parameters()).device

    @property
    def embedding_dimension(self) -> int:
        """
        Output embedding dimensionality.
        """

        return self.config.embedding_dim

    def trainable_parameters(self):
        """
        Iterator over trainable parameters.
        """

        return (
            p
            for p in self.parameters()
            if p.requires_grad
        )

    def num_trainable_parameters(self) -> int:
        """
        Number of trainable parameters.
        """

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def num_parameters(self) -> int:
        """
        Total parameter count.
        """

        return sum(
            p.numel()
            for p in self.parameters()
        )

    def extra_repr(self) -> str:

        return (
            f"embed_dim={self.config.embedding_dim}, "
            f"layers={self.config.num_layers}, "
            f"heads={self.config.num_heads}, "
            f"frozen={self.config.freeze_backbone}"
        )
        
# Internal forward helper

    def _forward_single(
        self,
        signal: torch.Tensor,
        mask: Optional[torch.Tensor]=None,
    ):
        """
        Forward one modality through the original SleepFM backbone.

        Parameters
        ----------
        signal
            Shape:
                (B, C, T)

        mask
            Shape:
                (B, C)

        Returns
        -------
        pooled_embedding
            (B, embedding_dim)

        sequence_embedding
            (B, sequence_length, embedding_dim)
        """

        signal = signal.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        pooled_embedding, sequence_embedding = self.backbone(signal,mask)
        return pooled_embedding, sequence_embedding
        
    def forward(self,
                modality_tensors: Union[Dict[str, torch.Tensor],List[torch.Tensor]],
                modality_masks: Optional[Union[Dict[str, torch.Tensor], List[torch.Tensor]]] = None, 
                ):
        """
        Forward every available modality.

        Returns
        -------
        dict

            {
                modality:
                {
                    "embedding": ...,
                    "sequence": ...
                }
            }
        """

        outputs = {}
        if isinstance(modality_tensors, list):
            items = enumerate(modality_tensors)
            get_mask = lambda idx: modality_masks[idx] if modality_masks is not None else None
        else:
            items = modality_tensors.items()
            get_mask = lambda key: modality_masks[key] if modality_masks is not None and key in modality_masks else None
        for key, tensor in items:
            mask = get_mask(key) 
            pooled, sequence = self._forward_single(tensor, mask)
            outputs[key] = {
                "embedding": pooled,
                "sequence": sequence,
            }

        return outputs
        
    @torch.no_grad()
    def encode(self,modality_tensors,modality_masks,):
        """
        Return pooled embeddings.

        This is the interface used by Strategy 1
        (feature extraction).
        """

        self.eval()

        outputs = self.forward(
            modality_tensors,
            modality_masks,
        )

        return {
            modality: out["embedding"]
            for modality, out in outputs.items()
        }   
    
    @torch.no_grad()
    def extract_tokens(self,signal: torch.Tensor,):
        """
        Return CNN token embeddings before
        any transformer layers.

        Shape

            (B,C,T)

        ->
            (B,C,S,E)
        """

        self.eval()

        signal = signal.to(self.device)

        return self.backbone.patch_embedding(signal)
        
    @torch.no_grad()
    def encode_mean(self,modality_tensors,modality_masks,):
        """
        Return one embedding for the entire recording.

        Multiple modality embeddings are averaged.
        """

        embeddings = self.encode(modality_tensors,modality_masks,)

        stacked = torch.stack(list(embeddings.values()),dim=0,)

        return stacked.mean(dim=0)
        
    @torch.no_grad()
    def encode_concat(self,modality_tensors,modality_masks,):
        """
        Concatenate all modality embeddings.

        Example

        EEG 128

        RESP 128

        ECG 128

        -->

        384 dimensional vector
        """

        embeddings = self.encode(
            modality_tensors,
            modality_masks,
        )

        ordered = []

        for modality in sorted(embeddings.keys()):
            ordered.append(embeddings[modality])

        return torch.cat(ordered,dim=-1,)
        
    def embedding_size(self,num_modalities: int = 1,aggregation: str = "mean",):
        """
        Return output dimensionality.
        """

        if aggregation == "mean":
            return self.embedding_dimension

        if aggregation == "concat":
            return (num_modalities* self.embedding_dimension)

        raise ValueError(f"Unknown aggregation '{aggregation}'.")
    
    
def load_sleepfm(config: SleepFMConfig, checkpoint_path: Optional[Union[str, Path]] = None) -> SleepFMEncoder:
    """
    Instantiates the SleepFMEncoder model, loads pre-trained weights via 
    SleepFMCheckpoint, and sets the model to evaluation mode.
    """
    model = SleepFMEncoder(config)
    
    path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_path
    if path and path.exists():
        ckpt_handler = SleepFMCheckpoint(config)
        model = ckpt_handler.load_backbone(model, checkpoint_path=path)
        
    model.to(config.device)
    model.eval()
    return model


def save_embeddings(embeddings: torch.Tensor, output_path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
    """
    Saves generated patient/sequence embeddings and associated metadata to disk.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "embeddings": embeddings.detach().cpu(),
        "metadata": metadata or {}
    }
    torch.save(payload, path)


def load_embeddings(embedding_path: Union[str, Path], device: str = "cpu") -> Tuple[torch.Tensor, Dict]:
    """
    Loads cached embeddings and metadata from a given file path.
    """
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found at: {path}")
        
    data = torch.load(path, map_location=device)
    return data["embeddings"], data.get("metadata", {})


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all parameters within the backbone model to prevent gradient updates.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_layers(model: nn.Module, n_layers: int) -> None:
    """
    Unfreezes the parameters of the last `n_layers` of a Transformer encoder.
    """
    if n_layers <= 0:
        return

    if hasattr(model, "transformer_encoder"):
        layers = list(model.transformer_encoder.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = list(model.transformer.layers)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        layers = list(model.encoder.layers)
    else:
        # Fallback: find all children modules named 'layer' or 'layers'
        layers = [m for name, m in model.named_modules() if "layer" in name.lower()]

    for layer in layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    
class FeatureExtractorStrategy:
    """
    Strategy 1: Frozen feature extraction strategy using SleepFM as a static embedding generator.
    """
    def __init__(self, config: SleepFMConfig):
        self.config = config
        self.preprocessor = SleepFMPreprocessor(config)
        self.model = load_sleepfm(config)
        freeze_backbone(self.model)

    def extract_features(self, modality_signals: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        tensors, masks = self.preprocessor.preprocess(modality_signals)
        with torch.no_grad():
            embeddings = self.model(tensors, masks)
        return embeddings.squeeze(0).cpu().numpy()

class FineTuneStrategy(nn.Module):
    """
    Strategy 2: End-to-end fine-tuning strategy with an optional prediction head.
    """
    def __init__(self, config: SleepFMConfig, num_classes: int = 2):
        super().__init__()
        self.config = config
        self.backbone = SleepFMEncoder(config)
        
        if config.freeze_backbone:
            freeze_backbone(self.backbone)
            if config.unfreeze_last_n_layers:
                unfreeze_last_layers(self.backbone, config.unfreeze_last_n_layers)
                
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dimension, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, 
                modality_tensors: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
                modality_masks: Optional[Union[Dict[str, torch.Tensor], List[torch.Tensor]]] = None
                ) -> torch.Tensor:
        outputs = self.backbone(modality_tensors, modality_masks)
        embeddings = [out["embedding"] for out in outputs.values()] 
        stacked = torch.stack(embeddings, dim=0).mean(dim=0)
        logits = self.classifier(stacked)
        return logits

class HybridStrategy(nn.Module):
    """
    Strategy 3: Fuses handcrafted domain features with SleepFM continuous embeddings.
    """
    def __init__(self, config: SleepFMConfig, handcrafted_dim: int, num_classes: int = 2):
        super().__init__()
        self.config = config
        self.backbone = load_sleepfm(config)
        
        if config.freeze_backbone:
            freeze_backbone(self.backbone)

        self.handcrafted_projection = nn.Sequential(
            nn.Linear(handcrafted_dim, config.handcrafted_projection_dimension),
            nn.BatchNorm1d(config.handcrafted_projection_dimension),
            nn.ReLU()
        )

        combined_dim = config.embedding_dimension + config.handcrafted_projection_dimension
        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, 
        modality_tensors: List[torch.Tensor], 
        handcrafted_features: torch.Tensor, 
        modality_masks: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        deep_embeddings = self.backbone(modality_tensors, modality_masks)
        hc_embeddings = self.handcrafted_projection(handcrafted_features)
        
        fused_vector = torch.cat([deep_embeddings, hc_embeddings], dim=-1)
        logits = self.fusion_classifier(fused_vector)
        return logits
    
