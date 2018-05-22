import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import normalize_labels, batch_indices

class CNN():
    '''
    Implements a convolutional neural network for classification.  
    
    activation:
        tf function, applied after each layer (except for the last one).
    epochs:
        Int, the number of training epochs.
    patch_sizes:
        A list, kernel widths for the convolutional layers. This list defines the number of the
        convlutional layers, too.
    n_filters:
        The number of feature maps for each convlutional layer. Must have the same length as the 'patch_sizes'.
    fully_connected_sizes:
        The sizes of the fully connected hidden layers (defines the number of fully connected layers)
    batch_size:
        Mini batch-size for the training.    
    learning_rate:
        Learning rate which is fed to 'tf.train.AdamOptimizer'.     
    validation_set_prop:
        Size of the validation set. A float between 0 and 1. 
        The Model is applied to this set after each epoch, and the accuracy is printed.        
    l2_penalty:
        Regularization parameter >= 0, l2-penalty is added to weight matrices (not bias terms).  
    dropout_keep:
        Dropout keep probability for the fully connected layers, if 1.0, no dropout is used.
    seed:
        Integer, to allow reproducible results.        
    '''
    def __init__(self, 
                 activation = tf.nn.relu, 
                 epochs = 10,
                 patch_sizes = [5,5],
                 n_filters = [32,16],
                 fully_connected_sizes = [256],
                 batch_size = 256,
                 learning_rate = 0.001,
                 validation_set_prop = 0.01,
                 l2_penalty = 1e-4,
                 dropout_keep = 1.0,
                 seed = 1236):
        
        if len(patch_sizes) != len(n_filters):
            raise ValueError("patch_sizes and n_filters should be of equal length!")    
        
        self.fully_connected_sizes =  fully_connected_sizes
        self.dropout = dropout_keep
        self.n_filters = n_filters
        self.activation = activation
        self.epochs = epochs
        self.patch_sizes = patch_sizes
        self.batch_size = batch_size
        self.epsss = 1e-32 # small constant added to logits before softmax
        self.lr = learning_rate
        self.seed = seed # this is used when choosing validation set and shuffling of batches and initilizing parameters
        self.valid_size = validation_set_prop
        self.l2 = l2_penalty
   

    def _size_after_final_layer(self,width,height, conv_layer_strides, max_pool_strides):
        '''
        Compute the dimension of the input tensor after a series of convolutional+maxpool layers.
        
        
        conv_layer_strides:
            a list containing the stride for each convolutional layer
        
        max_pool_strides:
            a list containing the stride for each max pool layer
        
        Assumes that stride is the same for the width and the height. 
        
        The above two lists are assumed to be of equal length, eq. max pool
        is always performed after convolution.
        '''
        n_layers = len(conv_layer_strides)
    
        for ii in range(n_layers):
            
            conv_stride = conv_layer_strides[ii]
            max_pool_stride = max_pool_strides[ii]
            
            width,height = int(np.ceil(width/conv_stride)), int(np.ceil(height/conv_stride))
            width,height = int(np.ceil(width/max_pool_stride)), int(np.ceil(height/max_pool_stride))
            
        return width, height
     
        
    def _parameters(self, 
                    width,
                    height, 
                    num_channels, 
                    patch_sizes,
                    n_filters, 
                    fully_connected_sizes,
                    n_labels,
                    **kwargs):
        '''
        Create tf variables for all the parameters of the network. 
        
        Returns two dictionaries from which the convolutional layer parameters 
        can be accessed via keys "conv1", "conv2", ...
        and the fully connected layer parameters with "fc1", "fc2",...
        '''
        
        weights = {}
        biases = {}
        
        #  parameters for convolutional layers
        for ii,f_size in enumerate(n_filters):
            name = "conv" + str(ii)
            
            patch_size = patch_sizes[ii]
            
            # in first lauyer number of feature maps is the number of channels
            if ii == 0:            
                shape = [patch_size,patch_size,num_channels,f_size]
            else:
                shape = [patch_size,patch_size,n_filters[ii-1],f_size]
    
    
            bias_shape = shape[-1] # one bias parameters per filter/feature map
            seed = self.seed + ii
            
            n_inputs = shape[0]**2*shape[2] # TODO check if initilization is sensible
            stddev = np.sqrt(2.0/n_inputs)
            
            init_values = tf.truncated_normal(shape = shape,
                                              stddev= stddev,
                                              seed = seed)
            
            weights[name] = tf.Variable(init_values, 
                                          dtype = tf.float32,
                                          name = name)
            
            biases[name] =  tf.Variable(tf.zeros(bias_shape),
                                          dtype = tf.float32,
                                          name = name + "_bias")
            
        # fully connected layers    
        for jj in range(len(fully_connected_sizes) + 1):
            name = "fc" + str(jj)
        
            # first layer after convolutions
            if jj == 0:
                
                # compute the input size for the hidden layer
                n_layers = len(n_filters)
                conv_strides = kwargs.get("conv_strides", n_layers*[1])
                max_pool_strides = kwargs.get("max_pool_strides", n_layers*[2]) 
                
                final_w, final_h = self._size_after_final_layer(width,
                                                                height, 
                                                                conv_strides, 
                                                                max_pool_strides)                
                in_dim = final_w*final_h*n_filters[-1]
            else:
                in_dim = fully_connected_sizes[jj-1]
            
            if jj == len(fully_connected_sizes):
                out_dim = n_labels
            else:    
                out_dim = fully_connected_sizes[jj]
            
            w_shape = (in_dim,out_dim)  
            bias_shape = out_dim
            stddev = np.sqrt(2.0/in_dim)
            seed = self.seed + 2*jj
            
            init_values = tf.truncated_normal(shape = w_shape,
                                              stddev= stddev,
                                              seed = seed)
                
            weights[name] = tf.Variable(init_values, 
                                          dtype = tf.float32,
                                          name = name)
            
            biases[name] =  tf.Variable(tf.zeros(bias_shape),
                                          dtype = tf.float32,
                                          name = name + "_bias")
                
        return weights, biases
            
                
    def _fully_connected_layer(self,
                               input_tensor, 
                               W,
                               bias,
                               activation,
                               dropout,
                               seed_for_dropout):
        '''
        Create a single fully connected layer.
        '''
        
        if activation is not None:
            out = activation(tf.matmul(input_tensor,W)+bias)
        else:
            out = tf.matmul(input_tensor,W)+bias
            
        
        if dropout < 1:
            out = tf.nn.dropout(out,keep_prob = dropout, seed = seed_for_dropout)
            
        return out
   
    def _conv_layer(self,
                    input_tensor, 
                    filterr, 
                    bias,
                    activation,
                    dropout, 
                    seed_for_dropout = None,
                    **kwargs):
        '''
        Create a single convolutional layer (convolution + bias, activation, maxpool) 
        with given activation and possibly apply dropout to maxpool layer. 
        '''
        conv_padding = kwargs.get("conv_padding", "SAME")
        conv_strides = kwargs.get("conv_strides", [1,1,1,1])
        max_pool_strides = kwargs.get("max_pool_strides", [1,2,2,1])
        max_pool_padding = kwargs.get("max_pool_padding", "SAME")
        ksize = kwargs.get("ksize", [1,2,2,1])

        conv = tf.nn.conv2d(input_tensor,filterr, strides = conv_strides, padding = conv_padding)
        conv_with_bias =tf.nn.bias_add(conv,bias)
        activated = activation(conv_with_bias)
        pooled = tf.nn.max_pool(activated, ksize = ksize, strides = max_pool_strides, padding = max_pool_padding )
        
        if dropout < 1:
            pooled = tf.nn.dropout(pooled,keep_prob = dropout, seed = seed_for_dropout)
            
        return pooled

        
    
    def _model(self, X, weights, biases, n_conv_layer,n_fc_layers, TRAIN = True):
        '''
            Creates the actual model (convolutional layers + hidden layers).
            
            X:
                Input tensor.        
            weights:
                Dictionary, Weight parameters for layers, created using '_parameters' function.
            biases:
                Dictionary, bias parameters for layers, created using '_parameters' function.
            n_conv_layer:
                Int, the number of convolutional layers.
            n_fc_layers:
                Int, the number of fully connected layers.
            TRAIN:
                Boolean, tells whether to apply dropout. Should be 'False' during prediction.
        '''     
        for ii in range(n_conv_layer):
            name = "conv" + str(ii)
            
            filterr = weights[name]
            bias = biases[name]
            
            X = self._conv_layer(X,filterr,bias,self.activation,1)    
            
        x_shape = X.get_shape().as_list() 
        X = tf.reshape(X,[-1, x_shape[1]*x_shape[2]*x_shape[3]])
        
        for ii in range(n_fc_layers + 1):
            name = "fc" + str(ii) 
            
            W = weights[name]
            bias = biases[name]
    
    
            # apply dropout only during training
            if TRAIN:
                dropout = self.dropout
                seed_for_dropout = self.seed + 3*ii
            else:
                dropout = 1
                seed_for_dropout = None
                
                
            if ii == n_fc_layers:
                activation = None
            else:
                activation = self.activation
                
            X = self._fully_connected_layer(X,
                                            W,
                                            bias,
                                            activation = activation,
                                            dropout = dropout, 
                                            seed_for_dropout = seed_for_dropout)
            
        return X
        
    def train(self, X, y, test_X = None, return_probs = False):
        '''
        Learns parameters. 
        X               input data as an numpy array with shape [n_obs,width,height,num_channels], dtype should be np.float32 (it is converted if not)
        y               labels for the rows of X. Class labels will be converted to integers starting from 0.
        test_X          new examples which are classfied with the trained model
        return_probs    whether to return arg_max class or class probabilities for the test data
        '''
        if X.dtype is not np.float32:
            X = X.astype(np.float32)
        
        # normalize labels
        y, encoder,decoder = normalize_labels(y)
        self.label_encoder = encoder
        self.label_decoder = decoder

        n_labels = len(np.unique(y))

        # split a validation set 
        X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_size, random_state= self.seed + 3)
        
        n,width,height,num_channels = X.shape
        
        tf_X = tf.placeholder(dtype=tf.float32,shape = (None,width,height,num_channels))
        tf_y = tf.placeholder(dtype=tf.int32,shape = (None,))
        
        # Create the parameters for the model
        weights, biases = self._parameters(width,
                                           height,
                                           num_channels,
                                           self.patch_sizes,
                                           self.n_filters, 
                                           self.fully_connected_sizes,
                                           n_labels)
        
        n_conv_layers = len(self.n_filters)
        n_fc_layers = len(self.fully_connected_sizes)
        logits = self._model(tf_X,weights,biases,n_conv_layers, n_fc_layers, TRAIN = True)
        
        # optimization and the outputs
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= tf_y,
                                                                            logits=logits + self.epsss), axis= 0)
        
        
        # penalize weight matrices (biases are not penalized)
        if self.l2 > 0:
            penalty = 0
            
            for w in weights:
                penalty += tf.nn.l2_loss(weights[w])
                
            loss += self.l2*penalty
        
        
        y_hat = tf.nn.softmax(logits= logits + self.epsss)
        arg_max_class = tf.argmax(y_hat,1,output_type=tf.int32)
        
        train_step = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(loss)
        corrects = tf.equal(tf.argmax(y_hat,1,output_type=tf.int32),tf_y)
        accuracy = tf.reduce_mean(tf.cast(corrects,dtype=tf.float32),axis = 0)
                 
        init = tf.global_variables_initializer() 
       
        with tf.Session() as sess:
            
            sess.run(init)
            
            # training
            for ii in range(self.epochs):
                
                train_loss = 0
                train_acc = 0
                
                all_batch_indices = batch_indices(n,self.batch_size,shuffle = True, seed = self.seed + 2*ii) 
                n_batches = len(all_batch_indices)
                
                for inx in all_batch_indices:
                    X_batch = X[inx,:]
                    y_batch = y[inx]
                    
                    _ , l, a = sess.run([train_step,loss,accuracy], feed_dict = {tf_X : X_batch, tf_y: y_batch})
                    train_loss += l/n_batches
                    train_acc += a/n_batches
                    
                print("Epoch: %3d" % (ii + 1),"Train accuracy: %.2f" % train_acc, "Loss: %.4f" % train_loss)
                
                if self.valid_size > 0:
                    
                    n_valid = X_valid.shape[0]
                    out = np.zeros(n_valid)
                    
                    if self.batch_size > n_valid:
                        valid_batch = n_valid
                    else:
                        valid_batch = self.batch_size
                    
                    all_batch_indices = batch_indices(n_valid,valid_batch,shuffle = False) 
                    n_batches = len(all_batch_indices)
                
                    for inx in all_batch_indices:
                        X_valid_batch = X_valid[inx,:]
                        y_valid_batch = y_valid[inx]
                    
                    
                        out[inx] = sess.run(arg_max_class, feed_dict = {tf_X : X_valid_batch, tf_y: y_valid_batch})
                    
                    print("Validation accuracy: %.2f" % np.mean(np.equal(out,y_valid)))
                
            
            out = None
            # possible test examples
            if test_X is not None:
                    
                if test_X.dtype is not np.float32:
                    test_X = test_X.astype(np.float32)
                    
                n_test = test_X.shape[0]
                
                all_batch_indices_test = batch_indices(n_test,self.batch_size,shuffle = False)
                
                out = np.zeros(n_test)
                
                for inx in all_batch_indices_test:
                    
                    X_test_batch = test_X[inx,:]
                    logits = self._model(X_test_batch, weights,biases,n_conv_layers, n_fc_layers, TRAIN = False)   
                            
                    if return_probs:
                        y_hat = tf.nn.softmax(logits= logits + self.epsss)
                        out[inx] = sess.run(y_hat, feed_dict = {tf_X : X_test_batch})
                    else:
                        
                        arg_max_class = tf.argmax(y_hat,1,output_type=tf.int32)
                        out[inx] = sess.run(arg_max_class, feed_dict = {tf_X : X_test_batch})
            
        return out
                        
if __name__ == "__main__":
    
    from utils import fmnist_train_test_split
    seed = 12123
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    X_train, y_train, X_test, y_test = fmnist_train_test_split()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    n, d = X_train.shape
    img_d = int(np.sqrt(d))
    X_train = np.reshape(X_train,(X_train.shape[0],img_d,img_d,1))
    X_test = np.reshape(X_test,(X_test.shape[0],img_d,img_d,1))
    
    cnn = CNN(epochs=20, n_filters= [64,32,16,8], patch_sizes= [5,5,5,5], fully_connected_sizes = [128,128], dropout_keep= 1)
    y_hat = cnn.train(X_train,y_train,X_test)
    
    print("")
    print("Test accuracy with fmnist: ", np.mean(np.equal(y_hat,y_test)))

