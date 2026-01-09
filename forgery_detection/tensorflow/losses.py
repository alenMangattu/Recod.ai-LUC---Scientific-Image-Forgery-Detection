"""
Loss Functions for Forgery Detection - TensorFlow/Keras
========================================================
Combined losses optimized for segmentation with class imbalance.

Includes:
- Dice Loss: Region-based overlap metric
- Focal Loss: Handles class imbalance
- Boundary Loss: Edge accuracy
- Combined Loss: Weighted combination
"""

import tensorflow as tf
from tensorflow import keras


def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss for segmentation.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask (sigmoid output)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value (1 - Dice coefficient)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return 1.0 - dice


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
    """
    Focal loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask (sigmoid output)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Weighting factor for positive class
    
    Returns:
        Focal loss value
    """
    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Binary cross entropy
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    # Focal weight
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = tf.pow(1 - pt, gamma)
    
    # Alpha weight (higher for positive class)
    alpha_weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    focal = alpha_weight * focal_weight * bce
    
    return tf.reduce_mean(focal)


def bce_loss(y_true, y_pred):
    """Standard binary cross entropy loss."""
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    return tf.reduce_mean(bce)


def boundary_loss(y_true, y_pred):
    """
    Boundary loss - penalizes errors at mask boundaries.
    Uses Sobel filters for edge detection.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
    
    Returns:
        Boundary dice loss
    """
    # Sobel filters
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    
    def get_boundary(mask):
        edge_x = tf.nn.conv2d(mask, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edge_y = tf.nn.conv2d(mask, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        boundary = tf.sqrt(tf.square(edge_x) + tf.square(edge_y) + 1e-8)
        return tf.cast(boundary > 0.1, tf.float32)
    
    pred_boundary = get_boundary(y_pred)
    true_boundary = get_boundary(y_true)
    
    # Dice loss on boundaries
    smooth = 1.0
    pred_flat = tf.reshape(pred_boundary, [-1])
    true_flat = tf.reshape(true_boundary, [-1])
    
    intersection = tf.reduce_sum(pred_flat * true_flat)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(pred_flat) + tf.reduce_sum(true_flat) + smooth)
    
    return 1.0 - dice


def combined_loss(y_true, y_pred, dice_w=1.0, focal_w=1.0, bce_w=0.5, boundary_w=0.3):
    """
    Combined loss for forgery detection.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        dice_w: Weight for dice loss
        focal_w: Weight for focal loss
        bce_w: Weight for BCE loss
        boundary_w: Weight for boundary loss
    
    Returns:
        Weighted sum of losses
    """
    loss = 0.0
    
    if dice_w > 0:
        loss += dice_w * dice_loss(y_true, y_pred)
    
    if focal_w > 0:
        loss += focal_w * focal_loss(y_true, y_pred)
    
    if bce_w > 0:
        loss += bce_w * bce_loss(y_true, y_pred)
    
    if boundary_w > 0:
        loss += boundary_w * boundary_loss(y_true, y_pred)
    
    return loss


class ForgeryLoss(keras.losses.Loss):
    """
    Custom Keras Loss class for forgery detection.
    Combines Dice, Focal, BCE, and Boundary losses.
    
    Args:
        dice_weight: Weight for Dice loss
        focal_weight: Weight for Focal loss
        bce_weight: Weight for BCE loss
        boundary_weight: Weight for Boundary loss
    """
    
    def __init__(self, dice_weight=1.0, focal_weight=1.0, bce_weight=0.5, 
                 boundary_weight=0.3, name='forgery_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
    
    def call(self, y_true, y_pred):
        return combined_loss(
            y_true, y_pred,
            dice_w=self.dice_weight,
            focal_w=self.focal_weight,
            bce_w=self.bce_weight,
            boundary_w=self.boundary_weight
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dice_weight': self.dice_weight,
            'focal_weight': self.focal_weight,
            'bce_weight': self.bce_weight,
            'boundary_weight': self.boundary_weight,
        })
        return config


class DeepSupervisionLoss(keras.losses.Loss):
    """
    Loss with deep supervision for multi-output models.
    Uses auxiliary outputs to improve gradient flow.
    
    Args:
        main_weight: Weight for main output loss
        aux_weights: Weights for auxiliary outputs
    """
    
    def __init__(self, main_weight=1.0, aux_weights=(0.4, 0.2), name='deep_supervision_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.main_weight = main_weight
        self.aux_weights = aux_weights
        self.base_loss = ForgeryLoss()
    
    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Ground truth mask
            y_pred: Tuple of (main_output, aux1, aux2) or single output
        """
        if isinstance(y_pred, (list, tuple)):
            main_out, aux1, aux2 = y_pred
            loss = self.main_weight * self.base_loss(y_true, main_out)
            loss += self.aux_weights[0] * self.base_loss(y_true, aux1)
            loss += self.aux_weights[1] * self.base_loss(y_true, aux2)
        else:
            loss = self.base_loss(y_true, y_pred)
        
        return loss


def compute_deep_supervision_loss(y_true, main_pred, ds4_pred, ds3_pred,
                                   main_weight=1.0, ds4_weight=0.4, ds3_weight=0.2):
    """
    Compute loss with deep supervision outputs.
    Use this function in custom training loops.
    
    Args:
        y_true: Ground truth mask
        main_pred: Main output prediction
        ds4_pred: Deep supervision output 4
        ds3_pred: Deep supervision output 3
        main_weight: Weight for main loss
        ds4_weight: Weight for ds4 loss
        ds3_weight: Weight for ds3 loss
    
    Returns:
        Total weighted loss
    """
    loss = main_weight * combined_loss(y_true, main_pred)
    loss += ds4_weight * dice_loss(y_true, ds4_pred)  # Only dice for aux outputs
    loss += ds3_weight * dice_loss(y_true, ds3_pred)
    
    return loss


# Metrics
def dice_coefficient(y_true, y_pred, threshold=0.5):
    """Compute Dice coefficient (F1 score for segmentation)."""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.0)


def iou_score(y_true, y_pred, threshold=0.5):
    """Compute IoU (Intersection over Union) score."""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_binary, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    return (intersection + 1.0) / (union + 1.0)


if __name__ == '__main__':
    # Test losses
    print("Testing loss functions...")
    
    # Create dummy data
    y_true = tf.random.uniform((2, 256, 256, 1), 0, 1)
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.random.uniform((2, 256, 256, 1), 0, 1)
    
    # Test individual losses
    print(f"Dice loss: {dice_loss(y_true, y_pred):.4f}")
    print(f"Focal loss: {focal_loss(y_true, y_pred):.4f}")
    print(f"BCE loss: {bce_loss(y_true, y_pred):.4f}")
    print(f"Boundary loss: {boundary_loss(y_true, y_pred):.4f}")
    print(f"Combined loss: {combined_loss(y_true, y_pred):.4f}")
    
    # Test Keras loss class
    loss_fn = ForgeryLoss()
    print(f"ForgeryLoss class: {loss_fn(y_true, y_pred):.4f}")
    
    # Test metrics
    print(f"Dice coefficient: {dice_coefficient(y_true, y_pred):.4f}")
    print(f"IoU score: {iou_score(y_true, y_pred):.4f}")
    
    print("Loss functions test passed!")
