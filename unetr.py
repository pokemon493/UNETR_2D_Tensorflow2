import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import keras.backend as K
from keras.layers import (Layer, BatchNormalization, LayerNormalization, Conv2D, Conv2DTranspose, Embedding, 
    Activation, Dense, Dropout, MultiHeadAttention, add, Input, concatenate, GlobalAveragePooling1D)
from keras.models import Model

def mlp(x, hidden_units, dropout_rate):
    if not isinstance(hidden_units, list): hidden_units = [hidden_units]
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

class Patches(Layer):
    '''
    [B, H, W, C] 
    -> [B, H/patch_size, W/patch_size, C*(patch_size^2)] 
    -> [B, H*W/(patch_size^2), C*(patch_size^2)]
    '''
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    '''
    Project the patches to projection_dim and introduce learnable positional embedding for patches
    '''
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embeding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embeding(positions)
        return encoded

def normalization(input_tensor, normalization, name=None):

    if normalization=='batch':
        return(BatchNormalization(name=None if name is None else name + '_batchnorm')(input_tensor))
    elif normalization=='layer':
        return(LayerNormalization(epsilon=1e-6, name=None if name is None else name + '_layernorm')(input_tensor))
    elif normalization=='group':
        return(tfa.layers.GroupNormalization(groups=8, name=None if name is None else name + '_groupnorm')(input_tensor))
    elif normalization == None:
        return input_tensor
    else:
        raise ValueError('Invalid normalization')

def conv_norm_act(input_tensor, filters, kernel_size , norm_type='batch', act_type='relu', dilation=1):
    '''
    Conv2d + Normalization(norm_type:str) + Activation(act_type:str)
    '''
    output_tensor = Conv2D(filters, kernel_size, padding='same', dilation_rate=(dilation, dilation), use_bias=False if norm_type is not None else True, kernel_initializer='he_normal')(input_tensor)
    output_tensor = normalization(output_tensor, normalization=norm_type)
    if act_type is not None: output_tensor = Activation(act_type)(output_tensor)

    return output_tensor

def conv2d_block(input_tensor, filters, kernel_size, 
                norm_type, use_residual, act_type='relu',
                double_features = False, dilation=[1, 1], name=None):

    x = Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation[0], use_bias=False, kernel_initializer='he_normal', name=None if name is None else name + '_conv2d_0')(input_tensor)
    x = normalization(x, norm_type, name=None if name is None else name + '_0')
    x = Activation(act_type, name=None if name is None else name + act_type + '_0')(x)

    if double_features:
        filters *= 2

    x = Conv2D(filters, kernel_size, padding='same', dilation_rate=dilation[1], use_bias=False, kernel_initializer='he_normal', name=None if name is None else name + '_conv2d_1')(x)
    x = normalization(x, norm_type, name=None if name is None else name + '_1')

    if use_residual:
        if K.int_shape(input_tensor)[-1] != K.int_shape(x)[-1]:
            shortcut = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', name=None if name is None else name + '_shortcut_conv2d')(input_tensor)
            shortcut = normalization(shortcut, norm_type, name=None if name is None else name + '_shortcut')
            x = add([x, shortcut])
        else:
            x = add([x, input_tensor])

    x = Activation(act_type, name=None if name is None else name + act_type + '_0')(x)

    return x

def deconv_conv_block(x,
                      filters_list: list,
                      kernel_size,
                      norm_type,
                      act_type,
                      ):
    '''
    Corresponding to the blue block in the UNETR architecture diagram
    '''
    for filts in filters_list:
        x = Conv2DTranspose(filts, 2, (2, 2), kernel_initializer='he_normal')(x)
        x = conv_norm_act(x, filts, kernel_size, norm_type, act_type)
    return x

def conv_deconv_block(x,
                      filters,
                      kernel_size,
                      norm_type,
                      use_residual,
                      act_type,
                      ):
    '''
    Corresponding to the yellow + green block in the UNETR architecture diagram
    '''
    x = conv2d_block(x, filters, kernel_size, norm_type, use_residual, act_type)
    x = Conv2DTranspose(filters // 2, 2, (2, 2), kernel_initializer='he_normal')(x)
    return x


def create_vit(x,
               patch_size,
               num_patches,
               projection_dim,
               num_heads,
               transformer_units,
               transformer_layers,
               dropout_rate,
               extract_layers,
               ):
    skip_connections = []

    # Create patches.
    patches = Patches(patch_size)(x)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for layer in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads, dropout=dropout_rate
        )(x1, x1)
        # Skip connection 1.
        x2 = add([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        # Skip connection 2.
        encoded_patches = add([x3, x2])
        if layer + 1 in extract_layers:
            skip_connections.append(encoded_patches)

    return skip_connections


def build_model(# ↓ Base arguments
                input_shape = (256, 256, 3),
                class_nums = 5,
                # ↓ ViT arguments
                patch_size = 16,
                projection_dim = 768,
                num_heads = 12,
                transformer_units = [2048, 768],
                transformer_layers = 12,
                extract_layers = [3, 6, 9, 12],
                dropout_rate = 0.1,
                # ↓ Conv arguments
                kernel_size = 3,
                conv_norm = 'batch',
                conv_act = 'relu',
                use_residual = False,
                # ↓ Other arguments
                show_summary = True,
                output_act = 'auto',
                ):
    '''
    input_shape: tuple, (height, width, channel) note that this network is used for 2D image segmentation tasks
    class_nums: int, output channels
    patch_size: int, image partition size
    projection_dim: int, pojection dimensions in ViT
    num_heads: int, nums of heads for MultiHeadAttention
    transformer_units: list, the number of hidden units of the MLP module in ViT, note that it is in the form of a list, should be [hidden units, projection_dims]
    transformer_layers: int, transformer stacking nums
    extract_layers: list, Determine which layers in ViT should be added to the "skip connection", the default is [3, 6, 9, 12]
    dropout_rate: float, dropout ratio for the ViT part
    kernel_size: int, kernel size for convolution block
    conv_norm: str, The normalization method of the convolutional layer, 'batch' or 'layer' or 'group'
    conv_act: str, activation function for convilution layer
    use_residual: bool, whether the convolution module uses residual connections
    show_summary: bool, whether to show the model overview
    output_act: str, The activation function of the output layer will be determined according to class_nums when 'auto'
      (i.e. 'sigmoid' for binary segmentation task, 'softmax' for multi-class segmentation task)
    '''
    
    z4_de_filts = 512
    z3_de_filts_list = [512]
    z2_de_filts_list = [512, 256]
    z1_de_filts_list = [512, 256, 128]
    z34_conv_filts = 512
    z23_conv_filts = 256
    z12_conv_filts = 128
    z01_conv_filts = 64
    if output_act == 'auto': output_act = 'sigmoid' if class_nums == 1 else 'softmax'

    assert input_shape[0] == input_shape[1] and input_shape[0] // patch_size
    num_patches = (input_shape[0] * input_shape[1]) // (patch_size ** 2)

    inputs = Input(input_shape)
    z0 = inputs

    z1, z2, z3, z4 = create_vit(z0, 
                                patch_size,
                                num_patches,
                                projection_dim,
                                num_heads,
                                transformer_units,
                                transformer_layers,
                                dropout_rate,
                                extract_layers)
    
    z1 = tf.reshape(z1, (-1, input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim)) # [B, H/16, W/16, projection_dim]
    z2 = tf.reshape(z2, (-1, input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim))
    z3 = tf.reshape(z3, (-1, input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim))
    z4 = tf.reshape(z4, (-1, input_shape[0] // patch_size, input_shape[1] // patch_size, projection_dim))

    z4 = Conv2DTranspose(z4_de_filts, 2, (2, 2), kernel_initializer='he_normal')(z4)
    z3 = deconv_conv_block(z3, z3_de_filts_list, kernel_size, conv_norm, conv_act)
    z3 = concatenate([z3, z4])
    z3 = conv_deconv_block(z3, z34_conv_filts, kernel_size, conv_norm, use_residual, conv_act)
    z2 = deconv_conv_block(z2, z2_de_filts_list, kernel_size, conv_norm, conv_act)
    z2 = concatenate([z2, z3])
    z2 = conv_deconv_block(z2, z23_conv_filts, kernel_size, conv_norm, use_residual, conv_act)
    z1 = deconv_conv_block(z1, z1_de_filts_list, kernel_size, conv_norm, conv_act)
    z1 = concatenate([z1, z2])
    z1 = conv_deconv_block(z1, z12_conv_filts, kernel_size, conv_norm, use_residual, conv_act)
    z0 = conv2d_block(z0, z01_conv_filts, kernel_size, conv_norm, use_residual, conv_act)
    z0 = concatenate([z0, z1])
    z0 = conv2d_block(z0, z01_conv_filts, kernel_size, conv_norm, use_residual, conv_act)

    outputs = Conv2D(class_nums, 1, activation=output_act)(z0)

    model = Model(inputs=inputs, outputs=outputs)

    if show_summary: model.summary()

    return model


if __name__ == '__main__':
    x = np.random.uniform(size=(1, 256, 256, 3))
    model = build_model(# ↓ Base arguments
                input_shape = (256, 256, 3),
                class_nums = 5,
                # ↓ ViT arguments
                patch_size = 16,
                projection_dim = 768,
                num_heads = 12,
                transformer_units = [2048, 768],
                transformer_layers = 12,
                extract_layers = [3, 6, 9, 12],
                dropout_rate = 0.1,
                # ↓ Conv arguments
                kernel_size = 3,
                conv_norm = 'batch',
                conv_act = 'relu',
                use_residual = False,
                # ↓ Other arguments
                show_summary = True,
                output_act = 'auto',)
    y = model(x)
    print(x.shape, y.shape)
