import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import normalize_labels, batch_indices

class MLP():
    '''
    Implements the multilayer perceptron for classification.
    
    activation:
        tensorflow function, applied after each layer except for the last, where softmax is used        
    hidden_sizes:
        list, specifies the number of nodes in each hidden layer. 
        The length of the list gives the number of the hidden layers, must be at least 1.
    epochs:
        The number of training epochs. 
    batch_size:
        Mini batch-size for the training.    
    learning_rate:
        Learning rate which is fed to 'tf.train.AdamOptimizer'.     
    validation_set_prop:
        Size of the validation set. A float between 0 and 1. 
        The Model is applied to this set after each epoch, and the accuracy is printed.        
    l2_penalty:
        Regularization parameter >= 0, l2-penalty is added to weight matrices (not bias terms).       
    seed:
        Integer, to allow reproducible results.
    '''
    def __init__(self, 
                 activation = tf.nn.relu, 
                 hidden_sizes = [256],
                 epochs = 1,
                 batch_size = 128,
                 learning_rate = 0.001,
                 validation_set_prop = 0.1,
                 l2_penalty = 1e-4,
                 seed = 1236):
        
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsss = 1e-32 # small constant added to logits before softmax
        self.lr = learning_rate
        self.seed = seed # this is used when choosing validation set and shuffling of batches and initilizing parameters
        self.valid_size = validation_set_prop
        self.l2 = l2_penalty
        
    def _parameters(self, d, n_labels):
        '''
        Create tf variables for the weights and biases of input, hidden and output layers. 
        Variables are named "w0, w1, ... " (weights) and "b0, b1, ..." (bias terms),
        and then returned in two dictionaries. The number and the sizes of hidden layers are determined
        by the constructor parameter 'hidden_sizes'.
        '''
        weights = {}
        biases = {}
        
        n_hidden = len(self.hidden_sizes)
        
        for ii in range(n_hidden + 1):
            
            w_name = "w" + str(ii)
            b_name = "b" + str(ii)
            
            # input layer
            if ii == 0:    
                input_size = d
                output_size = self.hidden_sizes[ii]
            
            # output layer
            elif ii == n_hidden:
                input_size = self.hidden_sizes[ii-1]
                output_size = n_labels
                
            # hidden layers    
            else:
                input_size = self.hidden_sizes[ii-1]
                output_size = self.hidden_sizes[ii]
            
            
            w_shape = (input_size,output_size)  
            b_shape = output_size
            stddev = np.sqrt(2.0/input_size)
            seed = self.seed + ii
            
            # initial values for the weights, truncated normal with stdev proportional to
            # 1 / sqrt(number of inputs to layer)
            init_values = tf.truncated_normal(shape = w_shape,
                                              stddev= stddev,
                                              seed = seed)
            
            weights[w_name] = tf.Variable(init_values, 
                                          dtype = tf.float32,
                                          name = w_name)
            
            biases[b_name] =  tf.Variable(tf.zeros(b_shape),
                                          dtype = tf.float32,
                                          name = b_name)
            
        return weights, biases

    def _model(self, X, weights, biases):
        '''
            Define the layers of the model.
        '''      
        for ii in range(len(self.hidden_sizes)):
            
            W = weights["w"+str(ii)]
            b = biases["b"+str(ii)]
            X = self.activation(tf.matmul(X,W) + b)
            
        # no activation for the final layer (softmax is applied later)
        logit = tf.matmul(X,weights["w"+str(ii+1)]) + biases["b"+str(ii+1)]
        
        return logit
        
    def train(self, X, y, test_X = None, return_probs = False):
        '''
        Learns parameters. 
        X               input features as an numpy array, dtype should be np.float32 (it is converted if not)
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
        
        n,d = X.shape
        
        tf_X = tf.placeholder(dtype=tf.float32,shape = (None,d))
        tf_y = tf.placeholder(dtype=tf.int32,shape = (None,))
        
        # define the model
        weights, biases = self._parameters(d,n_labels)
        logits = self._model(tf_X,weights,biases)
        
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
                    val_acc = sess.run(accuracy, feed_dict = {tf_X : X_valid, tf_y: y_valid})
                    print("Validation accuracy: %.2f" % val_acc)
                
                out = None
                
            # possible test examples
            if test_X is not None:
                
                 if test_X.dtype is not np.float32:
                     test_X = test_X.astype(np.float32) 
                 
                 if return_probs:
                     out = sess.run(y_hat, feed_dict = {tf_X : test_X})
                 else:
                     out = sess.run(arg_max_class, feed_dict = {tf_X : test_X})
            
        return out
                        
if __name__ == "__main__":
    
    from utils import bc_train_test_split, fmnist_train_test_split
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    seed = 12123
    l2 = 1e-3
    
    X_train, y_train, X_test, y_test = bc_train_test_split(seed=seed,test_prop= 0.4)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLP(epochs=50, 
              batch_size= 256, 
              hidden_sizes=[256,128], 
              validation_set_prop= 0,
              l2_penalty = l2)
    
    y_hat = mlp.train(X_train,y_train,X_test)
    
    print("")
    print("Test accuracy with bc: ", np.mean(np.equal(y_hat,y_test)))
    print("--------------------------------------------------------------------")
    
    X_train, y_train, X_test, y_test = fmnist_train_test_split()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLP(epochs=20, 
              batch_size= 256, 
              hidden_sizes=[256,256], 
              validation_set_prop= 0.01,
              l2_penalty = l2)
    
    y_hat = mlp.train(X_train,y_train,X_test)
    
    print("")
    print("Test accuracy with fmnist: ", np.mean(np.equal(y_hat,y_test)))