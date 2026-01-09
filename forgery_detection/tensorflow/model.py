"""
Forgery Detection Model - TensorFlow/Keras Implementation
==========================================================

This module builds a U-Net style segmentation model with an encoder backbone.

Important Kaggle constraint:
- Competition notebooks typically run with internet disabled, so attempting to
  download ImageNet weights at runtime can fail.

To support offline pretrained backbones, this module provides:
- `encoder_type="resnet101v2_keras"`: tf.keras.applications.ResNet101V2
  (optionally weights="imagenet" if available).
- `encoder_type="savedmodel"`: loads a Keras-compatible SavedModel from disk and
  taps intermediate activations as skip connections (best effort).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet101V2


def _try_get_layer(model: keras.Model, names: list[str]):
    for name in names:
        try:
            return model.get_layer(name)
        except Exception:
            continue
    return None


def _is_feature_map_tensor(x):
    # Expect NHWC with spatial dims.
    try:
        return len(x.shape) == 4 and x.shape[1] is not None and x.shape[2] is not None
    except Exception:
        return False


def _build_encoder_model_from_layer_names(base: keras.Model, layer_names: list[str]) -> keras.Model:
    outputs = []
    for name in layer_names:
        layer = base.get_layer(name)
        outputs.append(layer.output)
    return keras.Model(inputs=base.input, outputs=outputs, name="encoder")


def _guess_skip_layers_by_resolution(base: keras.Model, target_scales=(2, 4, 8, 16, 32)):
    """
    Best-effort fallback when layer names are unknown: pick feature maps whose
    spatial resolution roughly matches /2,/4,/8,/16,/32 of the input.
    """
    inp = base.input
    if inp.shape.rank != 4:
        raise ValueError("Encoder input must be rank-4 NHWC")
    in_h, in_w = inp.shape[1], inp.shape[2]
    if in_h is None or in_w is None:
        # Use dynamic shapes: can't reliably match; fall back to last N feature maps.
        feature_layers = []
        for layer in base.layers:
            try:
                out = layer.output
            except Exception:
                continue
            if _is_feature_map_tensor(out):
                feature_layers.append(layer.name)
        # Heuristic: take the last 5 feature maps
        return feature_layers[-5:]

    desired = {(in_h // s, in_w // s): s for s in target_scales}
    best = {s: None for s in target_scales}

    for layer in base.layers:
        try:
            out = layer.output
        except Exception:
            continue
        if not _is_feature_map_tensor(out):
            continue
        h, w = out.shape[1], out.shape[2]
        if h is None or w is None:
            continue
        key = (int(h), int(w))
        if key in desired:
            s = desired[key]
            best[s] = layer.name  # last match at this scale wins

    names = [best[s] for s in target_scales if best[s] is not None]
    if len(names) < 3:
        # Fallback: last 5 feature maps
        feature_layers = []
        for layer in base.layers:
            try:
                out = layer.output
            except Exception:
                continue
            if _is_feature_map_tensor(out):
                feature_layers.append(layer.name)
        return feature_layers[-5:]

    return names


def _load_savedmodel_as_keras(savedmodel_dir: str) -> keras.Model:
    """
    Load a SavedModel that is Keras-loadable (TF2-style SavedModel).
    Note: this can fail under Keras 3 depending on backend/environment.
    """
    return tf.keras.models.load_model(savedmodel_dir, compile=False)


def _build_savedmodel_encoder(
    savedmodel_dir: str,
    input_shape: tuple[int, int, int],
    trainable: bool = True,
) -> keras.Model:
    base = _load_savedmodel_as_keras(savedmodel_dir)

    # Some SavedModels load as a keras.Sequential with a single layer; ensure it has an input.
    if not hasattr(base, "input") or base.input is None:
        inp = keras.Input(shape=input_shape, name="input")
        out = base(inp)
        base = keras.Model(inp, out, name="savedmodel_backbone")

    base.trainable = trainable

    # Try common ResNetV2-ish candidates (best effort; varies by export).
    candidate_skip_names = [
        # TF-Slim/Hub-ish naming patterns (non-exhaustive)
        "resnet_v2_101/conv1",
        "resnet_v2_101/block1/unit_3/bottleneck_v2/add",
        "resnet_v2_101/block2/unit_4/bottleneck_v2/add",
        "resnet_v2_101/block3/unit_23/bottleneck_v2/add",
        "resnet_v2_101/block4/unit_3/bottleneck_v2/add",
        # Sometimes layers are indexed or wrapped; keep fallbacks
        "conv1",
        "block1",
        "block2",
        "block3",
        "block4",
    ]

    # Prefer known exact names if they exist.
    found = []
    for name in candidate_skip_names:
        layer = _try_get_layer(base, [name])
        if layer is not None:
            try:
                if _is_feature_map_tensor(layer.output):
                    found.append(layer.name)
            except Exception:
                continue

    if len(found) >= 5:
        skip_names = found[:5]
    else:
        skip_names = _guess_skip_layers_by_resolution(base)

    print(f"[INFO] SavedModel encoder skip layers: {skip_names}")
    return _build_encoder_model_from_layer_names(base, skip_names)


class ConvBnRelu(layers.Layer):
    """Conv2D + BatchNormalization + ReLU block."""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class AttentionGate(layers.Layer):
    """
    Attention gate for skip connections.
    Focuses decoder features on relevant regions of the skip connection.
    """
    
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.W_gate = layers.Conv2D(filters, 1, padding='same', use_bias=False)
        self.W_skip = layers.Conv2D(filters, 1, padding='same', use_bias=False)
        self.relu = layers.ReLU()
        self.psi = layers.Conv2D(1, 1, padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')
    
    def call(self, gate, skip, training=None):
        # Gate signal from decoder
        g = self.W_gate(gate)
        
        # Resize gate to match skip if needed
        if g.shape[1] != skip.shape[1] or g.shape[2] != skip.shape[2]:
            g = tf.image.resize(g, [skip.shape[1], skip.shape[2]], method='bilinear')
        
        # Skip signal
        x = self.W_skip(skip)
        
        # Combine and compute attention
        attention = self.relu(g + x)
        attention = self.psi(attention)
        attention = self.sigmoid(attention)
        
        return skip * attention


class DecoderBlock(layers.Layer):
    """Decoder block with upsampling, attention, and convolutions."""
    
    def __init__(self, filters, skip_filters, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.UpSampling2D(size=2, interpolation='bilinear')
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(skip_filters)
        self.conv1 = ConvBnRelu(filters)
        self.conv2 = ConvBnRelu(filters)
    
    def call(self, x, skip, training=None):
        # Upsample
        x = self.upsample(x)
        
        # Resize if needed
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = tf.image.resize(x, [skip.shape[1], skip.shape[2]], method='bilinear')
        
        # Apply attention to skip connection
        if self.use_attention:
            skip = self.attention(x, skip, training=training)
        
        # Concatenate and convolve
        x = layers.concatenate([x, skip], axis=-1)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        
        return x


class ForgeryDetector(Model):
    """
    U-Net with ResNet101V2 encoder for forgery detection.
    
    Features:
    - Pretrained ImageNet encoder (built-in, no external files)
    - Attention gates on skip connections
    - Deep supervision for better gradient flow
    
    Args:
        input_shape: Input image shape (H, W, C)
        use_attention: Whether to use attention gates
        freeze_encoder: Whether to freeze encoder weights initially
    """
    
    def __init__(
        self,
        input_shape=(384, 384, 3),
        use_attention=True,
        freeze_encoder=False,
        encoder_type: str = "resnet101v2_keras",
        encoder_savedmodel_dir: str | None = None,
        encoder_weights: str | None = "imagenet",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder_type = encoder_type

        if encoder_type == "savedmodel":
            if not encoder_savedmodel_dir:
                raise ValueError("encoder_savedmodel_dir is required when encoder_type='savedmodel'")
            self.encoder_model = _build_savedmodel_encoder(
                savedmodel_dir=encoder_savedmodel_dir,
                input_shape=input_shape,
                trainable=not freeze_encoder,
            )

            # Infer skip channel sizes from encoder outputs (best effort; can be dynamic)
            e1, e2, e3, e4, e5 = self.encoder_model.outputs[:5]
            skip_ch = [
                int(e1.shape[-1]) if e1.shape[-1] is not None else 64,
                int(e2.shape[-1]) if e2.shape[-1] is not None else 256,
                int(e3.shape[-1]) if e3.shape[-1] is not None else 512,
                int(e4.shape[-1]) if e4.shape[-1] is not None else 1024,
                int(e5.shape[-1]) if e5.shape[-1] is not None else 2048,
            ]
        else:
            # Build ResNet101V2 encoder (can optionally be pretrained if weights available)
            try:
                self.encoder = ResNet101V2(
                    include_top=False,
                    weights=encoder_weights,
                    input_shape=input_shape,
                )
            except Exception as e:
                # Kaggle competition notebooks often have no internet, so weights download can fail.
                # Fall back to random init but keep the run alive.
                print(f"[WARN] Could not load ResNet101V2 weights={encoder_weights!r}: {e}")
                print("[WARN] Falling back to weights=None (random init). Provide offline weights for best performance.")
                self.encoder = ResNet101V2(
                    include_top=False,
                    weights=None,
                    input_shape=input_shape,
                )

            # Freeze encoder if requested
            if freeze_encoder:
                for layer in self.encoder.layers[:100]:  # Freeze early layers
                    layer.trainable = False

            # Skip connection layer names for ResNet101V2
            skip_layer_names = [
                'conv1_conv',           # 64 filters, /2
                'conv2_block3_out',     # 256 filters, /4
                'conv3_block4_out',     # 512 filters, /8
                'conv4_block23_out',    # 1024 filters, /16
                'post_relu',            # 2048 filters, /32 (bottleneck)
            ]

            self.encoder_model = Model(
                inputs=self.encoder.input,
                outputs=[self.encoder.get_layer(name).output for name in skip_layer_names],
                name='encoder'
            )
            skip_ch = [64, 256, 512, 1024, 2048]
        
        # Decoder blocks. Use inferred skip channels when possible.
        # e1,e2,e3,e4,e5 channels ideally ~ 64,256,512,1024,2048.
        self.dec5 = DecoderBlock(512, skip_ch[3], use_attention=use_attention, name='dec5')  # bottleneck->512, skip=e4
        self.dec4 = DecoderBlock(256, skip_ch[2], use_attention=use_attention, name='dec4')  # ->256, skip=e3
        self.dec3 = DecoderBlock(128, skip_ch[1], use_attention=use_attention, name='dec3')  # ->128, skip=e2
        self.dec2 = DecoderBlock(64, skip_ch[0], use_attention=use_attention, name='dec2')   # ->64, skip=e1
        
        # Final upsampling and output
        self.final_upsample = layers.UpSampling2D(size=2, interpolation='bilinear')
        self.final_conv1 = ConvBnRelu(32, name='final_conv1')
        self.final_conv2 = ConvBnRelu(32, name='final_conv2')
        self.output_conv = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='output')
        
        # Deep supervision heads
        self.ds4_conv = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='ds4')
        self.ds3_conv = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='ds3')
    
    def call(self, inputs, training=None, return_deep=False):
        input_shape = tf.shape(inputs)
        target_h, target_w = input_shape[1], input_shape[2]
        
        # Encoder - get skip connections
        e1, e2, e3, e4, e5 = self.encoder_model(inputs, training=training)
        # e1: /2 (64 ch), e2: /4 (256 ch), e3: /8 (512 ch), e4: /16 (1024 ch), e5: /32 (2048 ch)
        
        # Decoder with skip connections
        d5 = self.dec5(e5, e4, training=training)  # /16, 512 ch
        d4 = self.dec4(d5, e3, training=training)  # /8, 256 ch
        d3 = self.dec3(d4, e2, training=training)  # /4, 128 ch
        d2 = self.dec2(d3, e1, training=training)  # /2, 64 ch
        
        # Final upsampling
        d1 = self.final_upsample(d2)  # /1
        d1 = self.final_conv1(d1, training=training)
        d1 = self.final_conv2(d1, training=training)
        
        # Main output
        out = self.output_conv(d1)
        out = tf.image.resize(out, [target_h, target_w], method='bilinear')
        
        if return_deep:
            # Deep supervision outputs
            ds4 = self.ds4_conv(d5)
            ds4 = tf.image.resize(ds4, [target_h, target_w], method='bilinear')
            ds3 = self.ds3_conv(d4)
            ds3 = tf.image.resize(ds3, [target_h, target_w], method='bilinear')
            return out, ds4, ds3
        
        return out
    
    def get_config(self):
        return {
            'input_shape': self.encoder.input_shape[1:],
            'use_attention': True,
        }


def build_model(
    input_shape=(384, 384, 3),
    use_attention=True,
    freeze_encoder=False,
    encoder_type: str = "resnet101v2_keras",
    encoder_savedmodel_dir: str | None = None,
    encoder_weights: str | None = "imagenet",
):
    """
    Factory function to build the forgery detector model.
    
    Args:
        input_shape: Tuple of (H, W, C)
        use_attention: Whether to use attention gates
        freeze_encoder: Whether to freeze encoder weights
    
    Returns:
        ForgeryDetector model instance
    """
    model = ForgeryDetector(
        input_shape=input_shape,
        use_attention=use_attention,
        freeze_encoder=freeze_encoder,
        encoder_type=encoder_type,
        encoder_savedmodel_dir=encoder_savedmodel_dir,
        encoder_weights=encoder_weights,
    )
    
    # Build the model by calling it once
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input, training=False)
    
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing ForgeryDetector...")
    
    model = build_model(input_shape=(256, 256, 3))
    
    # Test forward pass
    x = tf.random.normal((2, 256, 256, 3))
    out = model(x, training=False)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Test with deep supervision
    out, ds4, ds3 = model(x, training=True, return_deep=True)
    print(f"Deep supervision: main={out.shape}, ds4={ds4.shape}, ds3={ds3.shape}")
    
    # Count parameters
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    
    print("Model test passed!")
