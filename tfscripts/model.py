from copy import deepcopy
import numpy as np
import tensorflow as tf

from tfscripts import layers as tfs


class DenseNN(tf.keras.Model):
    """Dense Neural Network"""

    def __init__(
        self,
        input_shape,
        fc_sizes,
        use_dropout_list=False,
        activation_list="elu",
        use_batch_normalisation_list=False,
        use_residual_list=False,
        dtype="float32",
        verbose=False,
    ):
        """Dense NN Model

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        fc_sizes : list of int
            The number of nodes for each layer. The ith int denotes the number
            of nodes for the ith layer. The number of layers is inferred from
            the length of 'fc_sizes'.
        use_dropout_list : bool, optional
            Denotes whether to use dropout in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        activation_list : str or callable, optional
            The type of activation function to be used in each layer.
            If only one activation is provided, it will be used for all layers.
        use_batch_normalisation_list : bool or list of bool, optional
            Denotes whether to use batch normalisation in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        use_residual_list : bool or list of bool, optional
            Denotes whether to use residual additions in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        dtype : str, optional
            The float precision type.
        verbose : bool, optional
            If True, print additional information during setup.
        """
        super().__init__(dtype=dtype)

        self._input_shape = input_shape
        self.fc_sizes = fc_sizes
        self.use_dropout_list = use_dropout_list
        self.activation_list = activation_list
        self.use_batch_normalisation_list = use_batch_normalisation_list
        self.use_residual_list = use_residual_list

        tf_dtype = getattr(tf, dtype)

        # create variables for model trafo
        trafo_shape = (1, self._input_shape[1])
        y_out_shape = (1, self.fc_sizes[-1])
        self.trafo_model_initialized = False
        self.x_mean = self.add_weight(
            name="trafo_model_x_mean",
            shape=trafo_shape,
            initializer="zeros",
            trainable=False,
            dtype=tf_dtype,
        )
        self.x_std = self.add_weight(
            name="trafo_model_x_std",
            shape=trafo_shape,
            initializer="ones",
            trainable=False,
            dtype=tf_dtype,
        )
        self.y_mean = self.add_weight(
            name="trafo_model_y_mean",
            shape=y_out_shape,
            initializer="zeros",
            trainable=False,
            dtype=tf_dtype,
        )
        self.y_std = self.add_weight(
            name="trafo_model_y_std",
            shape=y_out_shape,
            initializer="ones",
            trainable=False,
            dtype=tf_dtype,
        )

        # create weights and layers
        self.fc_layers = tfs.FCLayers(
            input_shape=self._input_shape,
            fc_sizes=self.fc_sizes,
            use_dropout_list=self.use_dropout_list,
            activation_list=self.activation_list,
            use_batch_normalisation_list=self.use_batch_normalisation_list,
            use_residual_list=self.use_residual_list,
            weights_list=None,
            biases_list=None,
            max_out_size_list=None,
            float_precision=tf_dtype,
            name="fc_layer",
            verbose=verbose,
        )

        # explicitly add these variables to the module
        # variables created in sub-module are not found otherwise..
        self._fc_vars = self.fc_layers.variables

    def create_trafo_model(self, inputs, y_true):
        """Create trafo model

        Parameters
        ----------
        inputs : tf.Tensor or array_like
            The input data.
            Shape: [n_batch, n_inputs]
        y_true: tf.Tensor or array_like
            The labels for the input data.
            Shape: [n_batch, n_labels]
        """
        self.x_mean.assign(np.mean(inputs, axis=0, keepdims=True))
        self.x_std.assign(np.std(inputs, axis=0, keepdims=True))

        self.y_mean.assign(np.mean(y_true, axis=0, keepdims=True))
        self.y_std.assign(np.std(y_true, axis=0, keepdims=True))

        self.trafo_model_initialized = True

    def _check_trafo_model(self):
        """Check if Trafo model has been initialized

        Raises
        ------
        ValueError
            If trafo model has not been initialized
        """
        if not self.trafo_model_initialized:
            raise ValueError(
                "Trafo model is not yet configured. "
                "Run model.create_trafo_model() first"
            )

    def call(self, inputs, training=False, keep_prob=None):
        """Apply DenseNN

        Parameters
        ----------
        inputs : tf.Tensor or array_like
            The input data.
            Shape: [n_batch, n_inputs]
        training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        tf.Tensor
            The output of the NN.
        """
        self._check_trafo_model()

        # normalize input data
        inputs_trafo = (inputs - self.x_mean) / (1e-3 + self.x_std)

        output_trafo = self.fc_layers(
            inputs_trafo, is_training=training, keep_prob=keep_prob
        )[-1]

        # invert normalization of labels
        output = output_trafo * (1e-3 + self.y_std) + self.y_mean

        return output

    def save_weights(self, filepath, **kwargs):
        """Save Model weights

        Parameters
        ----------
        filepath : str
            Path to output file.
        **kwargs
            Keyword arguments passed on to tf.keras.Model.save_weights()
        """
        self._check_trafo_model()
        super().save_weights(filepath=filepath, **kwargs)

    def save(self, filepath, **kwargs):
        """Save Model

        Parameters
        ----------
        filepath : str
            Path to output file.
        **kwargs
            Keyword arguments passed on to tf.keras.Model.save()
        """
        self._check_trafo_model()
        super().save(filepath=filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        """Load Model weights

        Parameters
        ----------
        filepath : str
            Path to output file.
        **kwargs
            Keyword arguments passed on to tf.keras.Model.load_weights()
        """
        super().load_weights(filepath=filepath, **kwargs).expect_partial()
        self.trafo_model_initialized = True

    def load(self, **kwargs):
        """Load Model

        Parameters
        ----------
        **kwargs
            Keyword arguments passed on to tf.keras.Model.load()
        """
        self._check_trafo_model()
        super().load(**kwargs)
        self.trafo_model_initialized = True

    def get_config(self):
        """Get Configuration of DenseNN

        Returns
        -------
        dict
            A dictionary with all configuration settings. This can be used
            to serealize and deserealize the model.
        """
        config = {
            "input_shape": self._input_shape,
            "fc_sizes": self.fc_sizes,
            "use_dropout_list": self.use_dropout_list,
            "activation_list": self.activation_list,
            "use_batch_normalisation_list": self.use_batch_normalisation_list,
            "use_residual_list": self.use_residual_list,
            "dtype": self.dtype,
        }
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                config[key] = list(value)
        return deepcopy(config)


class DenseNNGaussian(DenseNN):
    """Dense Neural Network with Gaussian uncertainty prediction"""

    def __init__(
        self,
        fc_sizes_unc,
        use_dropout_list_unc=False,
        activation_list_unc="elu",
        use_batch_normalisation_list_unc=False,
        use_residual_list_unc=False,
        use_nth_fc_layer_as_input=None,
        min_sigma_value=1e-3,
        verbose=False,
        **kwargs
    ):
        """Gaussian Uncertainty NN (Dense NN Model)

        Parameters
        ----------
        fc_sizes_unc : list of int
            The number of nodes for each uncertainty layer. The ith int
            denotes the number of nodes for the ith layer. The number of
            layers is inferred from the length of 'fc_sizes_unc'.
        use_dropout_list_unc : bool, optional
            Denotes whether to use dropout in the uncertainty layers.
            If only a single boolean is provided, it will be used for all
            layers.
        activation_list_unc : str or callable, optional
            The activation function to be used in each uncertainty layer.
            If only one activation is provided, it will be used for all
            layers.
        use_batch_normalisation_list_unc : bool or list of bool, optional
            Denotes whether to use batch normalisation in the uncertainty
            layers. If only a single boolean is provided, it will be used for
            all layers.
        use_residual_list_unc : bool or list of bool, optional
            Denotes whether to use residual additions in the uncertainty
            layers. If only a single boolean is provided, it will be used
            for all layers.
        use_nth_fc_layer_as_input : None or int, optional
            If None, the same inputs as for the main NN will be used as
            input for the uncertainty network.
            If not None, the nth layer of the main NN will be used as input
            for the uncertainty network.
            If negative, the layer is counted from the last layer.
            For example, use_nth_fc_layer_as_input=-2 denotes the second last
            layer.
        min_sigma_value : float
            The lower bound for the uncertainty estimation.
            This is used to ensure robustness of the training.
        **kwargs
            Keyword arguments that are passed on to DenseNN initializer.
        """
        super().__init__(verbose=verbose, **kwargs)

        self.fc_sizes_unc = fc_sizes_unc
        self.use_dropout_list_unc = use_dropout_list_unc
        self.activation_list_unc = activation_list_unc
        self.use_batch_normalisation_list_unc = (
            use_batch_normalisation_list_unc
        )
        self.use_residual_list_unc = use_residual_list_unc
        self.use_nth_fc_layer_as_input = use_nth_fc_layer_as_input
        self.min_sigma_value = min_sigma_value

        if self.use_nth_fc_layer_as_input is not None:
            self._input_shape_unc = [
                -1,
                self.fc_sizes[self.use_nth_fc_layer_as_input],
            ]
        else:
            self._input_shape_unc = self._input_shape

        # create weights and layers for uncertainty network
        self.fc_layers_unc = tfs.FCLayers(
            input_shape=self._input_shape_unc,
            fc_sizes=self.fc_sizes_unc,
            use_dropout_list=self.use_dropout_list_unc,
            activation_list=self.activation_list_unc,
            use_batch_normalisation_list=self.use_batch_normalisation_list_unc,
            use_residual_list=self.use_residual_list_unc,
            weights_list=None,
            biases_list=None,
            max_out_size_list=None,
            float_precision=getattr(tf, self.dtype),
            name="fc_layer_unc",
            verbose=verbose,
        )

        # explicitly add these variables to the module
        # variables created in sub-module are not found otherwise..
        self._fc_vars = self.fc_layers_unc.variables

    def call(self, inputs, training=False, keep_prob=None):
        """Apply model

        Parameters
        ----------
        inputs : tf.Tensor or array_like
            The input data.
            Shape: [n_batch, n_inputs]
        training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        tf.Tensor
            The predicted values.
        tf.Tensor
            The predicted uncertainties.
        """

        self._check_trafo_model()

        # normalize input data
        inputs_trafo = (inputs - self.x_mean) / (1e-3 + self.x_std)

        # get outputs of main NN for value prediction
        main_output = self.fc_layers(
            inputs_trafo, is_training=training, keep_prob=keep_prob
        )

        # run uncertainty sub-nework
        if self.use_nth_fc_layer_as_input is not None:
            inputs_unc = main_output[self.use_nth_fc_layer_as_input]
        else:
            inputs_unc = inputs_trafo
        outputs_unc = self.fc_layers_unc(
            inputs_unc, is_training=training, keep_prob=keep_prob
        )[-1]

        # set initialized value to variance of 1,
        # which would be a correct initial guess if labels
        # are also normalized
        outputs_unc += 1.0

        # invert normalization of labels
        outputs = main_output[-1] * (1e-3 + self.y_std) + self.y_mean
        outputs_unc = outputs_unc * (1e-3 + self.y_std)

        # force positive value
        outputs_unc = tf.math.abs(outputs_unc) + self.min_sigma_value

        # return
        return outputs, outputs_unc

    def get_config(self):
        """Get Configuration of model

        Returns
        -------
        dict
            A dictionary with all configuration settings. This can be used
            to serealize and deserealize the model.
        """
        config = super().get_config()
        config.update(
            {
                "fc_sizes_unc": self.fc_sizes_unc,
                "use_dropout_list_unc": self.use_dropout_list_unc,
                "activation_list_unc": self.activation_list_unc,
                "use_batch_normalisation_list_unc": self.use_batch_normalisation_list_unc,
                "use_residual_list_unc": self.use_residual_list_unc,
                "use_nth_fc_layer_as_input": self.use_nth_fc_layer_as_input,
                "min_sigma_value": self.min_sigma_value,
            }
        )
        return deepcopy(config)
