
#-------------------------------------------------------------------
# conv4d equivalent with dilation
#-------------------------------------------------------------------
def conv4d_stacked(input, filter, 
                  strides=[1,1,1,1,1,1], 
                  padding='SAME',
                  dilation_rate=None,
                  stack_axis=None,
                  stack_nested=False,
                  ):
  '''
    Computes a convolution over 4 dimensions.
    Python generalization of tensorflow's conv3d with dilation.
    conv4d_stacked uses tensorflows conv3d and stacks results along
    stack_axis.

Parameters
----------
input : A Tensor.
        Shape [batch, x_dim, y_dim, z_dim, t_dim, in_channels]

filter: A Tensor. Must have the same type as input. 
        Shape [x_dim, y_dim, z_dim, t_dim, in_channels, out_channels]. 
        in_channels must match between input and filter

strides: A list of ints that has length 6. 1-D tensor of length 6. 
         The stride of the sliding window for each dimension of input. 
         Must have strides[0] = strides[5] = 1.
padding: A string from: "SAME", "VALID". The type of padding algorithm to use.

dilation_rate: Optional. Sequence of 4 ints >= 1. 
               Specifies the filter upsampling/input downsampling rate. 
               Equivalent to dilation_rate in tensorflows tf.nn.convolution

stack_axis: Int
          Axis along which the convolutions will be stacked.
          By default the axis with the lowest output dimensionality will be 
          chosen. This is only an educated guess of the best choice!

stack_nested: Bool
          If set to True, this will stack in a for loop seperately and afterwards 
          combine the results. In most cases slower, but maybe less memory needed.
      
Returns
-------
        A Tensor. Has the same type as input.
  ''' 
  # heuristically choose stack_axis
  if stack_axis == None:
    if dilation_rate == None:
      dil_array = np.ones(4)
    else:
      dil_array = np.asarray(dilation_rate)
    outputsizes = np.asarray(input.get_shape().as_list()[1:5])/np.asarray(strides[1:5])
    outputsizes -= dil_array*( np.asarray(filter.get_shape().as_list()[:4])-1)
    stack_axis = np.argmin(outputsizes)+1

  if dilation_rate != None:
    dilation_along_stack_axis = dilation_rate[stack_axis-1]
  else:
    dilation_along_stack_axis = 1

  tensors_t = tf.unstack(input,axis=stack_axis)
  kernel_t = tf.unstack(filter,axis=stack_axis-1)

  noOfInChannels = input.get_shape().as_list()[-1]
  len_ts = filter.get_shape().as_list()[stack_axis-1]
  size_of_t_dim = input.get_shape().as_list()[stack_axis]

  if len_ts % 2 ==1:
    # uneven filter size: same size to left and right
    filter_l = int(len_ts/2)
    filter_r = int(len_ts/2)
  else:
    # even filter size: one more to right
    filter_l = int(len_ts/2) -1
    filter_r = int(len_ts/2)

  # The start index is important for strides and dilation
  # The strides start with the first element
  # that works and is VALID:
  start_index = 0
  if padding == 'VALID':
    for i in range(size_of_t_dim):
      if len( range(  max(i - dilation_along_stack_axis*filter_l,0), 
                      min(i + dilation_along_stack_axis*filter_r+1,
                        size_of_t_dim),dilation_along_stack_axis)
            ) == len_ts:
        # we found the first index that doesn't need padding
        break
    start_index = i
    print 'start_index',start_index

  # loop over all t_j in t
  result_t = []
  for i in range(start_index,size_of_t_dim,strides[stack_axis]):

      kernel_patch = []
      input_patch = []
      tensors_t_convoluted = []

      if padding == 'VALID':

        # Get indices t_s
        indices_t_s = range(  max(i - dilation_along_stack_axis*filter_l,0), 
                              min(i + dilation_along_stack_axis*filter_r+1,size_of_t_dim), 
                              dilation_along_stack_axis)

        # check if Padding = 'VALID'
        if len(indices_t_s) == len_ts:

          # sum over all remaining index_t_i in indices_t_s
          for j,index_t_i in enumerate(indices_t_s):
            if not stack_nested:
              kernel_patch.append(kernel_t[j])
              input_patch.append(tensors_t[index_t_i])
            else:
              if dilation_rate != None:
                tensors_t_convoluted.append( tf.nn.convolution(input=tensors_t[index_t_i],
                                               filter=kernel_t[j],
                                               strides=strides[1:stack_axis+1]+strides[stack_axis:5],
                                               padding=padding,
                                               dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                                          )
              else:
                  tensors_t_convoluted.append( tf.nn.conv3d(input=tensors_t[index_t_i],
                                                   filter=kernel_t[j],
                                                   strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                   padding=padding)
                                              )
          if stack_nested:
            sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
            # put together
            result_t.append(sum_tensors_t_s)

      elif padding == 'SAME':

        # Get indices t_s
        indices_t_s = range(i - dilation_along_stack_axis*filter_l, 
                            (i + 1) + dilation_along_stack_axis*filter_r,
                            dilation_along_stack_axis)

        for kernel_j,j in enumerate(indices_t_s):
          # we can just leave out the invalid t coordinates
          # since they will be padded with 0's and therfore
          # don't contribute to the sum

          if 0 <= j < size_of_t_dim:
            if not stack_nested:
              kernel_patch.append(kernel_t[kernel_j])
              input_patch.append(tensors_t[j])
            else:
              if dilation_rate != None:
                  tensors_t_convoluted.append( tf.nn.convolution(input=tensors_t[j],
                                                 filter=kernel_t[kernel_j],
                                                 strides=strides[1:stack_axis+1]+strides[stack_axis:5],
                                                 padding=padding,
                                                 dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
                                            )
              else:
                  tensors_t_convoluted.append( tf.nn.conv3d(input=tensors_t[j],
                                                     filter=kernel_t[kernel_j],
                                                     strides=strides[:stack_axis]+strides[stack_axis+1:],
                                                     padding=padding)
                                                )
        if stack_nested:
          sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
          # put together
          result_t.append(sum_tensors_t_s)

      if not stack_nested:
        if kernel_patch:
          kernel_patch = tf.concat(kernel_patch,axis=3)
          input_patch = tf.concat(input_patch,axis=4)
          if dilation_rate != None:
              result_patch = tf.nn.convolution(input=input_patch,
                                         filter=kernel_patch,
                                         strides=strides[1:stack_axis]+strides[stack_axis+1:5],
                                         padding=padding,
                                         dilation_rate=dilation_rate[:stack_axis-1]+dilation_rate[stack_axis:])
          else:
              result_patch = tf.nn.conv3d(input=input_patch,
                                         filter=kernel_patch,
                                         strides=strides[:stack_axis]+strides[stack_axis+1:],
                                         padding=padding)
          result_t.append(result_patch)

  # stack together
  return tf.stack(result_t,axis=stack_axis)
















#-------------------------------------------------------------------
# Helper function to compute output length of a dimension
#-------------------------------------------------------------------
def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
  """Determines output length of a convolution given input length.

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "SAME", "VALID"
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).

  ADOPTED FROM:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
    contrib/keras/python/keras/utils/conv_utils.py
  """
  if input_length is None:
    return None
  assert padding in {'SAME', 'VALID'}

  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  if padding == 'SAME':
    output_length = input_length
  elif padding == 'VALID':
    output_length = input_length - dilated_filter_size + 1

  return (output_length + stride - 1) // stride


#-------------------------------------------------------------------
# Helper function to get filter steps to the left and right
#-------------------------------------------------------------------
def get_filter_lr(filter_size):
  '''
  Get the number of elements left and right
  of the filter position. If filtersize is
  even, there will be one more element to the
  right, than to the left.

  Arguments:
      filter_size: integer.

  Returns:
      Number of elemnts left and right 
      of the filter position. (int,int)

  '''
  if filter_size % 2 ==1:
    # uneven filter size: same size to left and right
    filter_l = filter_size // 2
    filter_r = filter_size // 2
  else:
    # even filter size: one more to right
    filter_l = max( (filter_size // 2) -1, 0)
    filter_r = filter_size // 2

  return filter_l, filter_r

#-------------------------------------------------------------------
# Helper function to compute start index for convolution of a dimension
#-------------------------------------------------------------------
def get_start_index(input_length, filter_size, padding, stride, dilation=1):
  '''
  Get start index for a convolution.
  This will be the index of the input axis
  for the first valid convolution position. 

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "SAME", "VALID"
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The start index for a convolution (integer).

  '''
  filter_l, filter_r = get_filter_lr(filter_size)

  # The start index is important for strides and dilation
  # The strides start with the first element
  # that works and is VALID:
  start_index = 0
  found_valid_position = False
  if padding == 'VALID':
    for i in range(input_length):
      if len( range(  max(i - dilation*filter_l, 0), 
                      min(i + dilation*filter_r + 1, input_length),
                      dilation)
            ) == filter_size:
        # we found the first index that doesn't need padding
        found_valid_position = True
        break

    if not found_valid_position:
      raise ValueError('Input dimension is too small for "VALID" patch')
    start_index = i

  return start_index

#-------------------------------------------------------------------
# Helper function to get slice for convolution patch
#-------------------------------------------------------------------
def get_conv_slice(position, input_length, filter_size, stride, dilation=1):
  '''
  Get slice for a convolution patch.
  The slice will correspond to the elements
  being convolved with the filter of size 
  filter_size.
  Note: slice gets cropped at 0 and maximum value 

  Arguments:
      position: current position of filter (int).
      input_length: integer.
      filter_size: integer.
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      slice of input being convolved with the filter
      at position. And number of elements cropped at
      either end. This is useful for padding
      (slice, (int,int))

  '''
  filter_l, filter_r = get_filter_lr(filter_size)

  min_index = position - dilation*filter_l
  max_index = position + dilation*filter_r + 1

  conv_slice =   slice( max(min_index, 0), 
                        min(max_index, input_length), 
                        dilation)

  padding_left = int( np.ceil( max( - min_index, 0) / dilation ) )
  padding_right = int( np.ceil( max( max_index - input_length, 0) / dilation ) )

  return conv_slice, (padding_left, padding_right)


#-------------------------------------------------------------------
# conv3d locally connected
#-------------------------------------------------------------------
def locally_connected_3d(input, 
                          num_outputs, 
                          filter_size,
                          strides=[1,1,1], 
                          padding='SAME', 
                          dilation_rate=None):
  '''
    Like conv3d, but doesn't share weights.

Parameters
----------
input:  A Tensor. Must be one of the following types: float32, float64, int64, int32, 
        uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half. 
        Shape [batch, in_depth, in_height, in_width, in_channels].

num_outputs: int
        Number of output channels

filter_size: list of int of size 3
            [filter x size, filter y size, filter z size]

strides:    A list of ints that has length >= 5. 1-D tensor of length 5. 
            The stride of the sliding window for each dimension of input.
            Must have strides[0] = strides[4] = 1.
padding:    A string from: "SAME", "VALID". The type of padding algorithm to use.

dilatrion_rate : None or list of int of length 3
            [dilattion in x, dilation in y, dilation in z]
            defines dilattion rate to be used

Returns
-------
        2 Tensors: result and kernels. 
        Have the same type as input.
  '''

  if dilation_rate == None:
    dilation_rate = [1,1,1]

  #------------------
  # get shapes
  #------------------
  input_shape = input.get_shape().as_list()

  # sanity checks
  assert len(filter_size) == 3, 'Filter size must be of shape [x,y,z], but is'.format(filter_size)
  assert np.prod(filter_size) > 0, 'Filter sizes must be greater than 0'
  assert len(input_shape) == 5, 'Shape is expected to be of length 5, but is {}'.format(input_shape)

  # calculate output shape
  output_shape = np.empty(5, dtype=int)
  for i in range(3):
    output_shape[i+1] = conv_output_length(input_length=input_shape[i + 1], 
                                filter_size=filter_size[i], 
                                padding=padding, 
                                stride=strides[i], 
                                dilation=dilation_rate[i])
  output_shape[0] = -1
  output_shape[4] = num_outputs

  num_inputs = input_shape[4]

  kernel_shape = ( np.prod(output_shape[1:-1]),
                   np.prod(filter_size) * num_inputs,
                   num_outputs
                 )

  #------------------
  # 1x1x1 convolution
  #------------------
  ## fast shortcut
  if list(filter_size) == [1,1,1]:
    kernel = new_weights(shape = input_shape[1:] + [num_outputs] )
    output = tf.reduce_sum( tf.expand_dims(input,axis=5) * kernel, axis = 4)
    return output, kernel


  #------------------
  # get slices
  #------------------
  start_indices = [ get_start_index(input_length=input_shape[i + 1], 
                                filter_size=filter_size[i], 
                                padding=padding, 
                                stride=strides[i], 
                                dilation=dilation_rate[i])
                    for i in range(3)]


  input_patches = []

  #---------------------------
  # loop over all x positions
  #---------------------------
  for x in range(start_indices[0], input_shape[1], strides[0]):

    # get slice for patch along x-axis
    slice_x, padding_x = get_conv_slice(x, 
                            input_length=input_shape[1], 
                            filter_size=filter_size[0], 
                            stride=strides[0], 
                            dilation=dilation_rate[0])
    
    if padding == 'VALID' and padding_x != (0,0):
      # skip this x position, since it does not provide
      # a valid patch for padding 'VALID'
      continue

    #---------------------------
    # loop over all y positions
    #---------------------------
    for y in range(start_indices[1], input_shape[2], strides[1]):

      # get indices for patch along y-axis
      slice_y, padding_y = get_conv_slice(y, 
                              input_length=input_shape[2], 
                              filter_size=filter_size[1], 
                              stride=strides[1], 
                              dilation=dilation_rate[1])

      if padding == 'VALID' and padding_y != (0,0):
        # skip this y position, since it does not provide
        # a valid patch for padding 'VALID'
        continue

      #---------------------------
      # loop over all z positions
      #---------------------------
      for z in range(start_indices[2], input_shape[3], strides[2]):

        # get indices for patch along y-axis
        slice_z, padding_z = get_conv_slice(z, 
                                input_length=input_shape[3], 
                                filter_size=filter_size[2], 
                                stride=strides[2], 
                                dilation=dilation_rate[2])

        if padding == 'VALID' and padding_z != (0,0):
          # skip this z position, since it does not provide
          # a valid patch for padding 'VALID'
          continue

        # At this point, slice_x/y/z either correspond
        # to a vaild patch, or padding is 'SAME'
        # Now we need to pick slice and add it to 
        # input patches. These will later be convolved 
        # with the kernel.

        #------------------------------------------
        # Get input patch at filter position x,y,z
        #------------------------------------------
        input_patch = input[:, slice_x, slice_y, slice_z, :]

        if padding == 'SAME':
          # pad with zeros 
          paddings = [(0,0),padding_x, padding_y, padding_z, (0,0)]
          if paddings != [(0,0),(0,0),(0,0),(0,0),(0,0)]:
            input_patch = tf.pad(input_patch,
                                  paddings= paddings,
                                  mode = 'CONSTANT',
                                  )

        # reshape
        input_patch = tf.reshape( input_patch, [-1, 1, np.prod(filter_size) * num_inputs, 1])

        # append to list
        input_patches.append(input_patch)
        #------------------------------------------

  # concat input patches
  input_patches = tf.concat(input_patches, axis=1)

  #------------------
  # get kernel
  #------------------
  kernel = new_weights(shape = kernel_shape)

  #------------------
  # perform convolution
  #------------------
  output = input_patches * kernel
  output = tf.reduce_sum(output, axis=2)
  output = tf.reshape(output, output_shape)
  return output, kernel
