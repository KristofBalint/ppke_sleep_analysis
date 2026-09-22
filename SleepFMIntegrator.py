from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Union
import numpy as np
from scipy.signal import resample
from SleepFM.models.models import SetTransformer
import torch
from torch import nn
from torch.utils.data import TensorDataset , DataLoader
'''
import os
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "SleepFMIntegrator.log")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
    force=True,
    # Important: force reconfiguration if logging was initialized elsewhere
)

logger = logging.getLogger(__name__)
'''

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
    #window_length_seconds: int = 5
    #window_stride_seconds: int = 5
    normalize_signals: bool = True
    normalization_method: str = "zscore"
    clip_value: Optional[float] = 8.0

    # Original SleepFM works on 5-minute recording chunks.
    sampling_duration_minutes: int = 5
    patch_size: int = 3840
    maximum_sequence_length: int = 128
    embedding_chunk_batch_size: int = 16
    # Patch tokenizer

    #patch_size: int = 640
    embedding_dimension: int = 128

    # Transformer

    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    pooling_heads: int = 4
    dropout: float = 0.10
    #maximum_sequence_length: int = 2048
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
            "BAS": 10,
            "RESP": 7,
            "EKG": 2,
            "EMG": 5,
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

    debug: bool = True
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
        self.chunk_samples = (config.sampling_duration_minutes * 60 * config.expected_sampling_frequency )
        self.patch_size = config.patch_size
        
        if self.chunk_samples % self.patch_size != 0:
            raise ValueError(
                f"SleepFM chunk length ({self.chunk_samples}) must be "
                f"divisible by patch_size ({self.patch_size})."
            )
        
        self.tokens_per_chunk = ( self.chunk_samples // self.patch_size )
        
        if self.tokens_per_chunk > config.maximum_sequence_length:
            raise ValueError(
                f"A SleepFM chunk creates {self.tokens_per_chunk} tokens, "
                f"but maximum_sequence_length="
                f"{config.maximum_sequence_length}."
            )

    # Public API

    def preprocess(self, modality_signals: Dict[str, Dict[str, np.ndarray]],
                   sampling_frequencies: Optional[Union[
                       int, Dict[str, Union[int, Dict[str, int]]]]] = None,
                   ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # def preprocess(self,modality_signals: Dict[str, Dict[str, np.ndarray]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
            else:
                mod_fs = sampling_frequencies
            tensor, mask = self._process_modality(modality, signals, mod_fs)

            modality_tensors.append(tensor)
            modality_masks.append(mask)

        return modality_tensors, modality_masks

    # Processing pipeline

    def _process_modality(self, modality: str, signals: Dict[str, np.ndarray],original_fs: Optional[Union[int, Dict[str, int]]] = None):
        processed = []
        for channel_name, signal in signals.items():
            chan_fs = original_fs.get(channel_name) if isinstance(original_fs,dict) else original_fs
            signal = self._validate_signal(signal)
            signal = self._resample(signal, original_fs=chan_fs)
            signal = self._normalize(signal)
            chunks = self.create_chunks(signal) 
            processed.append(chunks)

        tensor, mask = self._stack_channels(processed, modality)

        return tensor, mask

    # Validation

    def _validate_signal(self, signal: np.ndarray) -> np.ndarray:

        signal = np.asarray(signal).astype(np.float32)
        signal = np.nan_to_num(signal)
        if signal.ndim != 1:
            raise ValueError(
                "SleepFM only accepts 1D signals."
            )
        return signal

    # Resampling

    def _resample(self, signal: np.ndarray,original_fs: Optional[int] = None) -> np.ndarray:
        if original_fs is None or original_fs == self.fs:
            return signal
        target_length = int(len(signal) * self.fs / original_fs)
        return resample(signal, target_length)

    # Normalisation

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
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
            signal = np.clip(signal, -self.config.clip_value,
                             self.config.clip_value)

        return signal.astype(np.float32)

    # Windowing
    # The original had problems
    '''
    def create_windows(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) == 0:
            return np.empty((0, self.window_samples), dtype=np.float32)
        if len(signal) < self.window_samples:
            signal = np.pad(signal, (0, self.window_samples - len(signal)),
                            mode='constant')
        windows = []
        for start in range(0, len(signal) - self.window_samples + 1,
                           self.stride_samples):
            windows.append(signal[start:start + self.window_samples])
        if len(windows) == 0:
            return np.empty((0, self.window_samples), dtype=np.float32)
        return np.asarray(windows, dtype=np.float32)
    '''

    def create_chunks(self, signal: np.ndarray, ) -> np.ndarray:
        """
        Divide one channel into fixed-duration SleepFM chunks.
    
        Returns
        -------
        np.ndarray
            Shape:
                (num_chunks, chunk_samples)
    
        Notes
        -----
        SleepFM's SetTransformer tokenizer will itself divide each
        chunk into patch_size-sized temporal tokens. Therefore this
        function must NOT perform the 30-second patching itself.
        """
    
        signal = np.asarray( signal, dtype=np.float32, )
    
        if len(signal) == 0:
            return np.empty( (0, self.chunk_samples), dtype=np.float32, )
    
        # A recording shorter than one SleepFM chunk is padded toproduce one valid chunk.
        if len(signal) < self.chunk_samples:
            signal = np.pad( signal, ( 0, self.chunk_samples - len(signal), ), mode="constant", )
            return signal.reshape( 1, self.chunk_samples, )
    
        # Match the original SleepFM dataset behaviour:use complete chunks and discard the final incomplete part.
        num_chunks = ( len(signal) // self.chunk_samples )
        usable_samples = ( num_chunks * self.chunk_samples )
        signal = signal[:usable_samples]
        chunks = signal.reshape( num_chunks, self.chunk_samples, )
        return chunks.astype( np.float32, copy=False, )
        
    # Channel padding

    
    def _stack_channels(self, chunks_list: List[np.ndarray], modality: str, ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_channels = ( self.config.maximum_channels_per_modality[ modality ] )
        valid_chunks = [ x for x in chunks_list if ( isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] > 0 ) ]
        if not valid_chunks:
            # One dummy chunk.
            tensor = torch.zeros( ( 1, max_channels, self.chunk_samples, ), dtype=torch.float32, )
            mask = torch.ones( max_channels, dtype=torch.bool, )
            return tensor, mask
    
        # Respect maximum channel count.
        if len(valid_chunks) > max_channels:
            valid_chunks = valid_chunks[ :max_channels]
    
        # Different EDF channels can differ slightly in duration.
        # Only retain chunks present in every channel.
 
        num_chunks = min( x.shape[0] for x in valid_chunks )  
        valid_chunks = [ x[:num_chunks] for x in valid_chunks]
    
        # Current shape:
        #     (channels, chunks, samples)
        stacked = np.stack(valid_chunks, axis=0,)
        channels = stacked.shape[0]
    
        # Convert to:
        #     (chunks, channels, samples)
        stacked = np.transpose( stacked, (1, 0, 2), )
        padded = np.zeros( ( num_chunks, max_channels, self.chunk_samples, ), dtype=np.float32, )
        padded[:, :channels, :] = stacked
    
        mask = np.ones( max_channels, dtype=bool,)
        mask[:channels] = False   
    
        return ( torch.from_numpy(padded), torch.from_numpy(mask),)


class SleepFMCheckpoint:
    """
    Handles loading, saving and configuring pretrained SleepFM models.

    This class deliberately contains no preprocessing or inference logic.
    """

    def __init__(self, config: SleepFMConfig):
        self.config = config
        self.device = torch.device(config.device)

    # Checkpoint loading

    def load_backbone(self, model: nn.Module,checkpoint_path: Optional[Path] = None, ) -> nn.Module:
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        if checkpoint_path is None:
            raise ValueError("No checkpoint path specified.")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, )
        state_dict = self._extract_state_dict(checkpoint)
        incompatible = model.load_state_dict(state_dict,strict=self.config.strict_checkpoint_loading, )

        if self.config.verbose:
            self._print_loading_report(checkpoint_path, incompatible, )

        model.to(self.device)

        return model

    # Saving

    def save_checkpoint(self, model: nn.Module, output_path: Path,metadata: Optional[Dict] = None, ):
        output = {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        }
        torch.save(output, output_path, )

    # Extract state_dict

    def _extract_state_dict(self, checkpoint: Dict, ) -> Dict:
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        return checkpoint

    # Freeze

    def freeze_model(self, model: nn.Module, ):
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Unfreeze everything

    def unfreeze_model(self, model: nn.Module, ):
        for parameter in model.parameters():
            parameter.requires_grad = True

    # Unfreeze last transformer layers

    def unfreeze_last_layers(self, transformer: nn.Module, n_layers: int, ):
        if n_layers <= 0:
            return
        layers = list(transformer.transformer_encoder.layers)
        for layer in layers[-n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    # Freeze tokenizer only

    def freeze_tokenizer(self, model: nn.Module, ):

        if not hasattr(model, "patch_embedding"):
            return

        for parameter in model.patch_embedding.parameters():
            parameter.requires_grad = False

    # Unfreeze tokenizer

    def unfreeze_tokenizer(self, model: nn.Module, ):

        if not hasattr(model, "patch_embedding"):
            return

        for parameter in model.patch_embedding.parameters():
            parameter.requires_grad = True

    # Parameter summary

    def trainable_parameter_count(self, model: nn.Module, ) -> int:

        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    # Report

    def _print_loading_report(self, checkpoint_path: Path, incompatible, ):

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

    def __init__(self, config: SleepFMConfig, checkpoint_handler: Optional[SleepFMCheckpoint] = None, ) -> None:

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

        self.debug = getattr(config, "debug", True)

    # Backbone construction

    def _build_backbone(self) -> SetTransformer:
        """
        Construct the original SleepFM SetTransformer.

        This mirrors the architecture used during SleepFM pretraining.
        """

        backbone = SetTransformer(
            in_channels=self.config.maximum_channels_per_modality,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embedding_dimension,
            num_heads=self.config.num_attention_heads,
            num_layers=self.config.num_transformer_layers,
            pooling_head=self.config.pooling_heads,
            dropout=self.config.dropout,
            max_seq_length=self.config.maximum_sequence_length,
        )

        return backbone

    # Checkpoint handling

    def load_checkpoint(self, checkpoint: str | Path,strict: bool = False) -> None:
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

        return self.config.embedding_dimension

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
            f"embed_dim={self.config.embedding_dimension}, "
            f"layers={self.config.num_transformer_layers}, "
            f"heads={self.config.num_attention_heads}, "
            f"frozen={self.config.freeze_backbone}"
        )

    # Internal forward helper

    def _forward_single(self,signal: torch.Tensor,mask: Optional[torch.Tensor] = None,):
        """
        Encode one SleepFM modality.
        Parameters
        ----------
        signal
            Tensor with shape:
                (B, C, T)
            where B is the number of 5-minute chunks.
        mask
            Channel mask with shape:
                (C,)
            or
                (B, C)
            True indicates a padded channel.
        Returns
        -------
        pooled_embedding
            Shape:
                (B, embedding_dimension)
            One pooled SleepFM embedding per 5-minute chunk.
        sequence_embedding
            Shape:
                (B, S, embedding_dimension)
            S is normally 10 for 5-minute chunks with
            30-second patches.
        """
    
        signal = signal.to(device=self.device,dtype=self.config.embedding_dtype, )
    
        # Permit a single unbatched chunk.
        if signal.ndim == 2:
            signal = signal.unsqueeze(0)
    
        if signal.ndim != 3:
            raise ValueError(
                "SleepFM input must have shape "
                "(C,T) or (B,C,T). "
                f"Received {tuple(signal.shape)}."
            )
    
        B, C, T = signal.shape
    
        if T % self.config.patch_size != 0:
            raise ValueError(
                f"SleepFM chunk length {T} is not divisible "
                f"by patch_size={self.config.patch_size}."
            )
    
        sequence_length = ( T // self.config.patch_size )
        if (sequence_length > self.config.maximum_sequence_length ):
            raise ValueError(
                f"SleepFM chunk produces "
                f"{sequence_length} tokens, but "
                f"maximum_sequence_length="
                f"{self.config.maximum_sequence_length}."
            )
    
        self._debug(
            f"_forward_single: "
            f"B={B} chunks, C={C} channels, "
            f"T={T}, tokens/chunk={sequence_length}"
        )
    
        # Channel mask

        if mask is None:
            mask = torch.zeros((B, C), dtype=torch.bool,device=self.device, )
        else:
            mask = mask.to(device=self.device,dtype=torch.bool,)
            if mask.ndim == 1:
                if len(mask) != C:
                    raise ValueError(
                        f"Channel mask has {len(mask)} entries "
                        f"but signal has {C} channels."
                    )
                mask = (mask.unsqueeze(0).expand(B, -1))
            elif mask.ndim == 2:
                if mask.shape != (B, C):
                    raise ValueError(
                        f"Expected mask shape {(B, C)}, "
                        f"received {tuple(mask.shape)}."
                    )
            else:
                raise ValueError(
                    "SleepFM mask must have shape "
                    "(C,) or (B,C)."
                )
    
        # Avoid putting every 5-minute chunk through the CNN/
        # Transformer at once.

        batch_size = getattr(self.config,"embedding_chunk_batch_size",16,)
        pooled_parts = []
        sequence_parts = []
    
        for start in range(0,B,batch_size,):
            end = min(start + batch_size, B,)
            pooled, sequence = self.backbone(signal[start:end],mask[start:end],)
            pooled_parts.append(pooled)
            sequence_parts.append(sequence)
    
        pooled_embedding = torch.cat(pooled_parts,dim=0,)
        sequence_embedding = torch.cat(sequence_parts,dim=0,)
        return ( pooled_embedding,sequence_embedding,)

    def forward(self,
                modality_tensors: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
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
        self._debug("=" * 80)
        self._debug("SleepFMEncoder.forward()")
        self._debug(f"modality_tensors type: {type(modality_tensors)}")
        self._debug(f"modality_masks type: {type(modality_masks)}")

        if isinstance(modality_tensors, dict):
            self._debug(
                f"modality_tensors keys: {list(modality_tensors.keys())}"
            )

            for modality, tensor in modality_tensors.items():
                self._debug(
                    f"  {modality}: "
                    f"type={type(tensor)}, "
                    f"shape={tuple(tensor.shape)}, "
                    f"dtype={tensor.dtype}, "
                    f"device={tensor.device}"
                )

        elif isinstance(modality_tensors, list):
            self._debug(
                f"modality_tensors length: {len(modality_tensors)}"
            )

            for i, tensor in enumerate(modality_tensors):
                self._debug(
                    f"  [{i}]: "
                    f"type={type(tensor)}, "
                    f"shape={tuple(tensor.shape)}, "
                    f"dtype={tensor.dtype}, "
                    f"device={tensor.device}"
                )

        else:
            raise TypeError(
                "modality_tensors must be a dict or list, "
                f"got {type(modality_tensors)}"
            )

        if modality_masks is None:
            self._debug("modality_masks: None")

        elif isinstance(modality_masks, dict):
            self._debug(
                f"modality_masks keys: {list(modality_masks.keys())}"
            )

            for modality, mask in modality_masks.items():
                self._debug(
                    f"  {modality}: "
                    f"type={type(mask)}, "
                    f"shape={tuple(mask.shape)}, "
                    f"dtype={mask.dtype}, "
                    f"device={mask.device}"
                )

        elif isinstance(modality_masks, list):
            self._debug(
                f"modality_masks length: {len(modality_masks)}"
            )

            for i, mask in enumerate(modality_masks):
                self._debug(
                    f"  [{i}]: "
                    f"type={type(mask)}, "
                    f"shape={tuple(mask.shape)}, "
                    f"dtype={mask.dtype}, "
                    f"device={mask.device}"
                )

        else:
            raise TypeError(
                "modality_masks must be a dict, list, or None, "
                f"got {type(modality_masks)}"
            )

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
    def encode(self, modality_tensors, modality_masks, ):
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
    def extract_tokens(self, signal: torch.Tensor, ):
        """
        Return CNN token embeddings before
        any transformer layers.
        Shape
            (B,C,T) -> (B,C,S,E)
        """
        self.eval()
        signal = signal.to(self.device)
        return self.backbone.patch_embedding(signal)

    @torch.no_grad()
    def encode_mean(self, modality_tensors, modality_masks, ):
        """
        Return one embedding for the entire recording.
        Multiple modality embeddings are averaged.
        """

        embeddings = self.encode(modality_tensors, modality_masks, )
        stacked = torch.stack(list(embeddings.values()), dim=0, )
        return stacked.mean(dim=0)

    @torch.no_grad()
    def encode_concat(self, modality_tensors, modality_masks, ):
        """
        Concatenate all modality embeddings.
        Example
        EEG 128
        RESP 128
        ECG 128
        -->
        384 dimensional vector
        """

        embeddings = self.encode(modality_tensors,modality_masks,)

        ordered = []

        for modality in sorted(embeddings.keys()):
            ordered.append(embeddings[modality])

        return torch.cat(ordered, dim=-1, )

    def embedding_size(self, num_modalities: int = 1,aggregation: str = "mean", ):
        """
        Return output dimensionality.
        """

        if aggregation == "mean":
            return self.embedding_dimension

        if aggregation == "concat":
            return (num_modalities * self.embedding_dimension)

        raise ValueError(f"Unknown aggregation '{aggregation}'.")

    def _debug(self, message):
        if self.debug:
            print(f"[SleepFMEncoder DEBUG] {message}")
            #logger.info(f"[SleepFMEncoder DEBUG] {message}")


def load_sleepfm(config: SleepFMConfig, checkpoint_path: Optional[
    Union[str, Path]] = None) -> SleepFMEncoder:
    """
    Instantiates the SleepFMEncoder model, loads pre-trained weights via 
    SleepFMCheckpoint, and sets the model to evaluation mode.
    """
    model = SleepFMEncoder(config)

    path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_path
    if path is not None:
        if not path.exists():
            raise FileNotFoundError( f"SleepFM checkpoint not found: {path}")
    
        ckpt_handler = SleepFMCheckpoint(config)
        ckpt_handler.load_backbone(model.backbone,checkpoint_path=path,)

    model.to(config.device)
    model.eval()
    return model


def save_embeddings(embeddings: torch.Tensor, output_path: Union[str, Path],
                    metadata: Optional[Dict] = None) -> None:
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


def load_embeddings(embedding_path: Union[str, Path], device: str = "cpu") -> \
Tuple[torch.Tensor, Dict]:
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
        layers = [m for name, m in model.named_modules() if
                  "layer" in name.lower()]

    for layer in layers[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

@torch.no_grad()
def extract_sleepfm_patient_embedding(encoder,preprocessor,modality_signals,modality_fs,):
    """
    Produce one patient-level SleepFM vector by averaging the pooled
    embeddings from the actually present modalities.
    Missing/all-padded modalities do not participate in the mean.
    """

    encoder.eval()
    tensors, masks = preprocessor.preprocess(modality_signals,sampling_frequencies=modality_fs,)
    outputs = encoder(tensors,masks,)
    valid_embeddings = []

    for idx, output in outputs.items():
        mask = masks[idx]
        if bool(mask.all()):
            continue
        chunk_embeddings = output["embedding"]
        # One SleepFM representation for this modality
        # across the full night.
        modality_embedding = (chunk_embeddings.mean(dim=0,keepdim=True,))
        valid_embeddings.append(modality_embedding)
    
    if not valid_embeddings:
        raise RuntimeError("No valid SleepFM modalities were available.")
    
    # Mean across BAS / RESP / EKG / EMG.
    deep_embedding = torch.stack(valid_embeddings,dim=0,).mean(dim=0)
    
    return (deep_embedding.squeeze(0).cpu().numpy().astype(np.float32))
#implementations of different strategies
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
            combined_tensor = self.model.encode_concat(tensors, masks)
        return combined_tensor.squeeze(0).cpu().numpy()


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
                unfreeze_last_layers(self.backbone,
                                     config.unfreeze_last_n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dimension, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self,
                modality_tensors: Union[
                    Dict[str, torch.Tensor], List[torch.Tensor]],
                modality_masks: Optional[
                    Union[Dict[str, torch.Tensor], List[torch.Tensor]]] = None
                ) -> torch.Tensor:
        outputs = self.backbone(modality_tensors, modality_masks)
        embeddings = [out["embedding"] for out in outputs.values()]
        stacked = torch.stack(embeddings, dim=0).mean(dim=0)
        logits = self.classifier(stacked)
        return logits


class HybridStrategy(nn.Module):
    """
    Strategy 3: Hybrid handcrafted + SleepFM model.

    SleepFM processes each modality independently and returns one
    pooled embedding per modality. These modality embeddings are
    fused into a single patient-level SleepFM representation.

    The patient-level SleepFM representation is then concatenated
    with a learned projection of the handcrafted features.

    Flow
    ----
        modality tensors
            I
            V
        SleepFMEncoder
            I
            V
        one embedding per modality
            I
            V
        modality fusion
            I
        SleepFM patient embedding
                                I
                                V
                         I---handcrafted features
                         I    I  
                         I    V
                         I   projection
                         I     I
        SleepFM embedding -----I
                         I
                         V
                    concatenation
                         I
                         V
                  fusion classifier
                         I
                         V
                       logits
    """

    def __init__(self,config: SleepFMConfig,handcrafted_dim: int, num_classes: int = 2,):
        super().__init__()

        self.config = config

        # SleepFM backbone
        
        self.backbone = SleepFMEncoder(config)

        if config.freeze_backbone:
            freeze_backbone(self.backbone)

        # Handcrafted feature branch
  
        self.handcrafted_projection = nn.Sequential(
            nn.Linear(
                handcrafted_dim,
                config.handcrafted_projection_dimension,
            ),
            nn.LayerNorm(
                config.handcrafted_projection_dimension
            ),
            nn.ReLU(),
        )

        # Fusion classifier
 
        combined_dim = (
            config.embedding_dimension
            + config.handcrafted_projection_dimension
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        modality_tensors: Union[Dict[str, torch.Tensor],List[torch.Tensor],],
        handcrafted_features: torch.Tensor,
        modality_masks: Optional[Union[Dict[str, torch.Tensor],List[torch.Tensor],]] = None,
    ) -> torch.Tensor:

        # SleepFM
   
        sleepfm_outputs = self.backbone(
            modality_tensors,
            modality_masks,
        )

        # Extract pooled embedding from each modality
 
        modality_embeddings = []

        for key, output in sleepfm_outputs.items():

            embedding = output["embedding"]

            modality_embeddings.append(embedding)

        if len(modality_embeddings) == 0:
            raise RuntimeError(
                "SleepFMEncoder returned no modality embeddings."
            )
        # Fuse modalities

        # Each embedding has shape:
        #     (B, embedding_dimension)
        # Stack:
        #     (num_modalities, B, embedding_dimension)
        # Mean:
        #     (B, embedding_dimension)

        stacked_embeddings = torch.stack(
            modality_embeddings,
            dim=0,
        )

        deep_embedding = stacked_embeddings.mean(dim=0)

        # Handcrafted branch
 
        handcrafted_embedding = self.handcrafted_projection(
            handcrafted_features
        )

        # Hybrid fusion
        fused_vector = torch.cat(
            [
                deep_embedding,
                handcrafted_embedding,
            ],
            dim=-1,
        )

        # Prediction
   
        logits = self.fusion_classifier(
            fused_vector
        )

        return logits
        
    def forward_from_embeddings(self,
        deep_embedding: torch.Tensor,
        handcrafted_features: torch.Tensor,
        ) -> torch.Tensor:

        handcrafted_embedding = self.handcrafted_projection(
            handcrafted_features
        )
    
        fused_vector = torch.cat(
            [deep_embedding, handcrafted_embedding],
            dim=-1,
        )
    
        return self.fusion_classifier(fused_vector)
        
def train_hybrid_head(model,handcrafted_matrix,sleepfm_matrix,labels,config,epochs=30,):
    device = torch.device(config.device)
    hc_tensor = torch.tensor(handcrafted_matrix,dtype=torch.float32,)
    sleepfm_tensor = torch.tensor(sleepfm_matrix,dtype=torch.float32,)
    y_tensor = torch.tensor(labels,dtype=torch.float32,).reshape(-1, 1)
    dataset = TensorDataset(hc_tensor, sleepfm_tensor, y_tensor,)
    loader = DataLoader(dataset,batch_size=min(config.batch_size, len(dataset)),shuffle=True,)
    
    model.to(device)
    model.train()
    
    trainable_parameters = [p for p in model.parameters() if p.requires_grad        ]
    
    optimizer = torch.optim.AdamW( trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay,)
    
    num_positive = float(np.sum(labels == 1))
    num_negative = float(np.sum(labels == 0))
    
    if num_positive > 0:
        pos_weight = torch.tensor( [num_negative / num_positive], device=device, )
    
        criterion = nn.BCEWithLogitsLoss( pos_weight=pos_weight )
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
    
        running_loss = 0.0
    
        for hc_batch, sleepfm_batch, y_batch in loader:
    
            hc_batch = hc_batch.to(device)
            sleepfm_batch = sleepfm_batch.to(device)
            y_batch = y_batch.to(device)
        
            optimizer.zero_grad(set_to_none=True)
        
            logits = model.forward_from_embeddings( sleepfm_batch, hc_batch,)
            loss = criterion( logits, y_batch,)
            loss.backward()
        
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_( trainable_parameters, config.gradient_clip_norm, )
        
            optimizer.step()
        
            running_loss += ( loss.item() * len(hc_batch) )
    
        if config.verbose:
            mean_loss = running_loss / len(dataset)
            print(
                f"Hybrid epoch {epoch + 1}/{epochs}, "
                f"loss={mean_loss:.5f}"
            )
    
    model.eval() 
    
    return model
