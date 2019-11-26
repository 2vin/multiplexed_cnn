from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import tf


class Multiplexer(Layer):
    def __init__(self, output_dim, nb_ctrl_sig, **kwargs):
        """
        This layer is used to split the output of a previous Dense layer into
        nb_ctrl_sig groups of size output_dim, and choose which group to provide
        as output using a discrete control signal.
        It takes as input two tensors, namely the output of the previous layer 
        and a column tensor with int32 or int64 values for the control signal.

        The Dense input to this layer must be of shape (None, prev_output_dim),
        where prev_output_dim = output_dim * nb_ctrl_sig.
        No checks are done at runtime to ensure that the input to the layer is
        correct, so it's better to double check.

        An example usage of this layer may be:

            input = Input(shape=(3,))
            control = Input(shape=(1,), dtype='int32')
            hidden = Dense(6)(i)  # output_dim == 2, nb_ctrl_sig == 3
            output = Multiplexer(2, 3)([hidden, control])
            model = Model(input=[input, control], output=output)
            
            ...
            
            x = randn(3)  # Input has size 3
            ctrl = array([0, 1, 2])
            
            # Outputs the first two neurons of the Dense layer
            model.predict([x, ctrl[0]])
            
            # Outputs the middle two neurons of the Dense layer
            model.predict([x, ctrl[1]])
            
            # Outputs the last two neurons of the Dense layer
            model.predict([x, ctrl[2]])
            
        # Arguments
            output_dim: positive integer, dimensionality of the output space.
            nb_ctrl_sig: positive integer, number of groups in which to split 
                the output of the previous layer. Must satisfy the relation:
                input_size = nb_ctrl_sig * output_dim
        
        """
        self.output_dim = output_dim
        self.nb_ctrl_sig = nb_ctrl_sig
        super(Multiplexer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Multiplexer, self).build(input_shape)

    def call(self, args, mask=None):
        return self.multiplexer(args, self.output_dim, self.nb_ctrl_sig)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    @staticmethod
    def multiplexer(args, output_size, nb_actions):
        """
        Returns a tensor of shape (None, output_size) where each sample is
        the result of masking each sample in full_input with a binary mask that 
        preserves only output_size elements, based on the corresponding control 
        value in indices.
        """
        full_input, indices = args

        '''
        For example, given:
            full_input: [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
            nb_actions: 3
            output_size: 2
            indices: [[0], [2]]
            desired output: [[1, 2], [11, 12]]
        we want to output the first two elements (index 0) of the first sample 
        and the last two elements (index 2) of the second sample.
        To do this, we need the absolute indices [[0, 1], [4, 5]].

        To build these, first compute the base absolute indices (0 and 4) by
        multiplying the control indices for the output size:
            [[0], [2]] * 2 = [[0], [4]]

        '''
        base_absolute_indices = tf.multiply(indices, output_size)

        '''
        Build an array containing the base absolute indices repeated output_size
         times:
            [[0, 0], [4, 4]]
        '''
        bai_repeated = tf.tile(base_absolute_indices, [1, output_size])

        '''
        Finally, add range(output_size) to these tensors to get the full
        absolute indices:
            [0, 0] + [0, 1] = [0, 1]
            [4, 4] + [0, 1] = [4, 5]
        so we have:
            [[0, 1], [4, 5]]
        '''
        absolute_indices = tf.add(bai_repeated, tf.range(output_size))

        '''
        Flatten this tensor in order to compute the one hot encoding for each 
        absolute index:
            [0, 1, 4, 5]
        '''
        ai_flat = tf.reshape(absolute_indices, [-1])

        '''
        Compute the one-hot encoding for the absolute indices.
        From [0, 1, 4, 5] we get:
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
        '''
        ai_onehot = tf.one_hot(ai_flat, output_size * nb_actions)

        '''
        Build the mask for full_input from the one-hot-encoded absolute indices.
        We need to group the one-hot absolute indices into groups of output_size
        elements.
        We get:
            [
              [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
              [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
            ]
        '''
        group_shape = [-1, output_size, output_size * nb_actions]
        group = tf.reshape(ai_onehot, group_shape)

        '''
        Reduce_sum along axis 1 to collapse the group and get the binary masks.
            [[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]]
        '''
        masks = tf.reduce_sum(group, axis=1)

        '''
        Convert the mask to boolean.
            [[True, True, False, False, False, False],
             [False, False, False, False, True, True]]
        '''
        zero = tf.constant(0, dtype=tf.float32)
        bool_masks = tf.not_equal(masks, zero)

        '''
        Convert the boolean masks back to absolute indices for the full_input
        tensor (each element represents [sample index, value index]).
        We get:
            [[0, 0], [0, 1], [1, 4], [1, 5]]
        '''
        ai_mask = tf.where(bool_masks)

        '''
        Apply the masks to full_input. We get a 1D tensor:
            [1, 2, 11, 12]
        '''
        reduced_output = tf.gather_nd(full_input, ai_mask)

        '''
        Reshape the reduction to match the output shape.
        We get:
            [[1, 2], [11, 12]]
        '''
        return tf.reshape(reduced_output, [-1, output_size])

