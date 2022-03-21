
backend = None
layers = None
models = None
keras_utils = None
from keras_applications import get_submodules_from_kwargs



# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------
def my_FPNBlock_list(pyramid_filters, stage,input_tensor, used_skip,used_idx):
    conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
    conv1_name = 'fpn_stage_p{}_conv'.format(stage)
    add_name = 'fpn_stage_p{}_add'.format(stage)
    up_name = 'fpn_stage_p{}_upsampling'.format(stage)

    channels_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        # if input tensor channels not equal to pyramid channels
        # we will not be able to sum input tensor and skip
        # so add extra conv layer to transform it
    input_filters = backend.int_shape(input_tensor)[channels_axis]
    if input_filters != pyramid_filters:
        input_tensor = layers.Conv2D(
            filters=pyramid_filters,
            kernel_size=(1, 1),
            kernel_initializer='he_uniform',
            name=conv0_name,
        )(input_tensor)

    skip=mergex(used_skip,used_idx)


    skip = layers.Conv2D(
        filters=pyramid_filters,
        kernel_size=(1, 1),
        kernel_initializer='he_uniform',
        name=conv1_name,
    )(skip)

    # print(pyramid_filters, skip)

    x = layers.UpSampling2D((2, 2), name=up_name)(input_tensor)
    x = layers.Add(name=add_name)([x, skip])

    return x

def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper
block_id=100
def myattantionMulty(x):
    global block_id
    block_id += 1

    all_w = layers.GlobalAvgPool2D()(x)
    denseshape = backend.int_shape(all_w)[1]
    #
    all_w = layers.Dense(denseshape/2)(all_w)
    all_w= layers.Activation('relu')(all_w)
    all_w = layers.Dense(denseshape)(all_w)
    all_w = layers.Activation('sigmoid')(all_w)
    #
    shape_skip = backend.int_shape(x)
    all_w_ = layers.Reshape((1, 1, denseshape))(all_w)

    all_w_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=1),
                                 arguments={'repnum': shape_skip[1]})(all_w_)

    all_w_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=2),
                                 arguments={'repnum': shape_skip[2]})(all_w_repeat)

    print(backend.int_shape(x)[3])
    all_w_repeat = layers.Activation('sigmoid')(all_w_repeat)

    all = layers.multiply([x, all_w_repeat])
    print(backend.int_shape(all))

    block_id += 2

    shape_skip = backend.int_shape(x)
    skip__ = layers.Conv2D(1, (1, 1), padding='same')(x)
    # concat = layers.Concatenate(axis=concat_axis, name=concat_name)([x__, skip__])
    skip__ = layers.Activation('sigmoid', name='sig_' + str(block_id))(skip__)
    my_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=3),
                              arguments={'repnum': shape_skip[3]})(
        skip__)
    y = layers.multiply([my_repeat, x])

    output=layers.Maximum()([all, y])
    #output=layers.Add()([all, y])

    return output


def myattantion(x):
    global block_id
    block_id+=1

    all_w= layers.GlobalAvgPool2D()(x)
    denseshape=backend.int_shape(all_w)[1]
    #
    all_w = layers.Dense(denseshape/2)(all_w)
    all_w= layers.Activation('relu')(all_w)
    all_w = layers.Dense(denseshape)(all_w)
    all_w= layers.Activation('sigmoid')(all_w)
    #
    shape_skip = backend.int_shape(x)
    all_w_ = layers.Reshape((1, 1, denseshape))(all_w)

    all_w_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=1),
                                 arguments={'repnum': shape_skip[1]})(all_w_)

    all_w_repeat = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=2),
                                 arguments={'repnum': shape_skip[2]})(all_w_repeat)



    print(backend.int_shape(x)[3])
    all_w_repeat= layers.Activation('sigmoid')(all_w_repeat)


    all = layers.multiply([x, all_w_repeat])
    print(backend.int_shape(all))
    return all



def mergex(used_skip,used_idx):
    if len(used_skip)==1:
        return used_skip[0][used_idx]
    mergelist=[]

    for idx in range(len(used_skip)):
        mergelist.append(used_skip[idx][used_idx])

    x=layers.Concatenate(3)(mergelist)
    return myattantion(x)


# def build_fpn(
#
#         backbone=backbone,
#         classes=classes,
#         activation=activation,
#
#
#
#         bones,
#         backbone_name,
#         pyramid_filters=256,
#         segmentation_filters=128,
#         classes=1,
#         activation='sigmoid',
#         use_batchnorm=True,
#         aggregation='sum',
#         dropout=None,
#         richnet=True,
# inputs=None
# ):
def build_fpn(

        backbone=None,
        pyramid_filters=256,
        segmentation_filters=128,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        richnet=True,
        inputs=None
):

    used_skip=[]

    sk = []
    myskip=[]
    skip_connection_layers = mergepart["inceptionv3"][:4]
    for idx_layer,layer_con in enumerate(backbone.layers):
        # print(layer_con.name)
        if layer_con.name in skip_connection_layers or idx_layer in skip_connection_layers:
            sk.append(layer_con.output)
            if len(sk)>1:
                birles=layers.Concatenate(3)(sk)
                # birles=myattantion(birles)
                myskip.append(birles)
            else:
                myskip.append(sk[0])
            sk=[]
        elif 'mixed' in layer_con.name  or idx_layer in [6,13]:
            sk.append(layer_con.output)


        if 'mixed10' in layer_con.name:
            x=layers.Concatenate(3)(sk)
            # x=myattantion(x)
    myskip.append(x)
    skips_insep=[]
    for i in range(len(myskip)):
        skips_insep.append(myskip[len(myskip)-i-1])

    used_skip.append(skips_insep)
    x=mergex(used_skip,0)

    p5 = my_FPNBlock_list(pyramid_filters, 5, x, used_skip,1 )
    p4 = my_FPNBlock_list(pyramid_filters, 4,p5,used_skip,2 )
    p3 = my_FPNBlock_list(pyramid_filters, 3,p4,used_skip,3 )
    p2 = my_FPNBlock_list(pyramid_filters, 2,p3, used_skip,4 )

    # add segmentation head to each
    s5 = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage5')(p5)
    s4 = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage4')(p4)
    s3 = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage3')(p3)
    s2 = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage2')(p2)

    # upsampling to same resolution
    s5 = layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)
    s4 = layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)
    s3 = layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)

    x = layers.Concatenate(axis=3, name='aggregation_concat')([s2, s3, s4, s5])


    # x=myattantion(x)
    # x = SA_attantion(x)
    global Onemi

    if richnet:
        x=myattantionMulty(x)
    # final stage
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='final_stage')(x)
    x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsampling')(x)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    model = models.Model(backbone.input, x)

    return model

mergepart = {
        'inceptionv3': (228, 86, 16, 9)

    }

# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------
import tensorflow as tf
def Inc_ZOEA_MODEL(
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        type='bos',
        **kwargs
):
    _KERAS_BACKEND = tf.keras.backend
    _KERAS_LAYERS = tf.keras.layers
    _KERAS_MODELS = tf.keras.models
    _KERAS_UTILS = tf.keras.utils

    kwargs['backend'] = _KERAS_BACKEND
    kwargs['layers'] = _KERAS_LAYERS
    kwargs['models'] = _KERAS_MODELS
    kwargs['utils'] = _KERAS_UTILS

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # if key not in ['backend', 'layers', 'models', 'utils']:

    # backbone = tf.keras.applications.InceptionV3(
    #     include_top=False,
    #     weights="imagenet",
    #     input_shape=input_shape
    # )

    from segmentation_models.backbones import inception_v3 as iv3

    backbone = iv3.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
            **kwargs,
    )

    model = build_fpn(
        backbone=backbone,
        classes=classes,
        activation=activation,
    )


    return model


