from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D
from keras.layers.merge import Concatenate
import keras.activations as activations
from libs.non_local import non_local_block


class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras        
        param input_shape: list of dimensions for [img, mask]
        """
        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
            
        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        #*********************************************************************************************
        # Define kernel shape ofr the new-designed Inception modolue.              [WNY 2020.2 Canada]
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        kernel_size_small1=(self.kernel_size[0]-2,self.kernel_size[1]-2)       
        kernel_shape_small1=kernel_size_small1 + (self.input_dim, self.filters)
        kernel_size_small2=(self.kernel_size[0]-4,self.kernel_size[1]-4)       
        kernel_shape_small2=kernel_size_small2 + (self.input_dim, self.filters)
        # kernel_size_NIN=self.kernel_size
        # kernel_size_NIN[0]=1
        # kernel_size_NIN[1]=1
        # kernel_shape_NIN=kernel_size_NIN + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_small1 = self.add_weight(shape=kernel_shape_small1,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel_small1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_small2 = self.add_weight(shape=kernel_shape_small2,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel_small2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # self.kernel_NIN = self.add_weight(shape=kernel_shape_NIN,
        #                               initializer=self.kernel_initializer,
        #                               name='img_kernel_NIN',
        #                               regularizer=self.kernel_regularizer,
        #                               constraint=self.kernel_constraint)
        # This is the end of the definiton of Inception module.                     [WNY 2020.2 Canada]
        #**********************************************************************************************

        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        ''' 

        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Use ReLU to optimize mask_output.[WNY 2019 Canada]
        # mask_output= activations.relu(mask_output, alpha=0.0, max_value=None, threshold=int(self.window_size/2)+1)

        # Apply convolutions to image
        #*********************************************************************************************
        # Replace the original convolution with the new-designed Inception module.[WNY 2020.2 Canada]
        img_output_ori = K.conv2d(
            (images*masks), self.kernel, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        img_output_small1 = K.conv2d(
            (inputs[0]*inputs[1]), self.kernel_small1,
            strides=self.strides,
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        img_output_small2 = K.conv2d(
            (inputs[0]*inputs[1]), self.kernel_small2,
            strides=self.strides,
            padding='same',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        img_output =(img_output_ori+ img_output_small1+ img_output_small2)/3
        
        # add non-local-block here:
        if self.filters==128:
            img_output = non_local_block(img_output, compression=2, mode='dot', add_residual=False)
            #print("this layer's filter amount is :"+str(self.filters))
        # This is the end of the definiton of Inception module.                     [WNY 2020.2 Canada]
        #**********************************************************************************************

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)
        
        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]
