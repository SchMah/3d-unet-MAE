import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from config import cfg

def conv_block_3d(x, filters):
    x = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = Activation("relu")(x)
    
    x = Conv3D(filters, (3, 3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.GroupNormalization(groups=8)(x)
    x = Activation("relu")(x)
    return x

def build_3d_model(input_shape=cfg.PATCH_SIZE + (4,), num_classes=cfg.NUM_CLASSES, mode='segmentation'):
    """
    mode: 'mae_pretrain' (outputs MRI reconstruction) or 'segmentation' (outputs tumor mask).
    """
    inputs = Input(shape=input_shape)

    c1 = conv_block_3d(inputs, 16)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = conv_block_3d(p1, 32)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = conv_block_3d(p2, 64)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = conv_block_3d(p3, 128)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    c5 = conv_block_3d(p4, 256)

    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block_3d(u6, 128)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block_3d(u7, 64)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block_3d(u8, 32)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block_3d(u9, 16)

    if mode == 'mae_pretrain':
        outputs = Conv3D(4, (1, 1, 1), activation="linear", name="reconstruction_head")(c9)
    else:
        outputs = Conv3D(num_classes, (1, 1, 1), activation="softmax", name="segmentation_head")(c9)

    model = Model(inputs=inputs, outputs=outputs)
    return model