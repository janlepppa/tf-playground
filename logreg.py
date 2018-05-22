import tensorflow as tf
import numpy as np
from utils import normalize_labels
         
class LogReg():
    '''
    Implements (multiple) Logistic Regression using Tensorflow.
    '''  
    def __init__(self,
                 bias =  True, 
                 learning_rate = 0.001, 
                 iters = 250,
                 l2_penalty = 0):
        '''
        bias            Boolean. Whether to include intercept in the model.
        learning_rate   Non-negative scalar. Learning rate which is passed to AdamOptimizer.
        iters           Positive integer. The number of optimization steps/epochs.
        l2_penalty      Regularization >= 0. Adds l2_penalty/2*||W||**2 regularization term to loss function.
                        Bias terms are not penalized.
        '''
        self.bias = bias
        self.lr = learning_rate
        self.iters = iters
        self.W = None
        self.b = None
        self.losses = None
        self.l2 = l2_penalty
        self.epsss = 1e-32 # small constant added to logits before softmax
        
    def fit(self, X, y):
        '''
        Learns parameters.
        X       input features as an numpy array, dtype should be np.float32 (it is converted if not)
        y       labels for the rows of X. Class labels will be converted to integers starting from 0.
        '''
        if X.dtype is not np.float32:
            X = X.astype(np.float32)
        
        y, encoder,decoder = normalize_labels(y)
        self.label_encoder = encoder
        self.label_decoder = decoder
                
        n,d = X.shape
        n_labels = len(np.unique(y))
            
        tf_X = tf.placeholder(tf.float32, shape= (n,d))
        tf_y = tf.placeholder(tf.int32,shape= [n])
            
        W = tf.Variable(tf.zeros((d,n_labels)),dtype = tf.float32)
    
        if self.bias:
            b = tf.Variable(tf.zeros(n_labels),dtype = tf.float32)     
            unscaled_log_prop = tf.matmul(tf_X,W) + b
        else:           
            unscaled_log_prop = tf.matmul(tf_X,W)
         

        y_hat = tf.nn.softmax(logits= unscaled_log_prop + self.epsss)   
           
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= tf_y,
                                                                            logits=unscaled_log_prop + self.epsss), 
                                                                            axis= 0)
        
        if self.l2 > 0:
            l2_reg = self.l2*tf.nn.l2_loss(W)
            loss = loss + l2_reg
            
        
        train_step = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(loss)
        corrects = tf.equal(tf.argmax(y_hat,1,output_type=tf.int32),tf_y)
        accuracy = tf.reduce_mean(tf.cast(corrects,dtype=tf.float32),axis = 0)
        
        init = tf.global_variables_initializer()
        self.losses = []
        
        with tf.Session() as sess:
            sess.run(init)
 
            for ii in range(self.iters):
                inputs = {tf_X : X, tf_y : y}
                train_step.run(feed_dict= inputs)
                
                self.losses.append(loss.eval(inputs))
                
                if (ii + 1) % 50 == 0:
                    print("Iterations: %3d" % (ii+1) ,"Train accuracy: %.2f" % accuracy.eval(inputs), "Loss: %.4f" % self.losses[-1])
                                              
            self.W = W.eval()
            self.b = None
            
            if self.bias:
                self.b = b.eval()
                    
    def predict(self, X, output_probs = False):
        '''
        Returns predicted probabilities or argmax class for test observations.
        
        X               test data
        output_probs    Boolean. If 'False' returns predicted class for each observation.
        '''
        if self.W is None:
            raise TypeError("No weights found. Run .fit() to learn the parameters.")
        if self.bias == True and self.b is None:
             raise TypeError("No bias term found. Run .fit() to learn the parameters.") 
             
        if X.dtype is not np.float32:      
            tf_X = tf.constant(X.astype(np.float32))
        else:
            tf_X = tf.constant(X)  
            
        W = tf.constant(self.W,dtype = tf.float32)
        
        if self.bias:
            b = tf.Variable(self.b,dtype = tf.float32)     
            unscaled_log_prop = tf.matmul(tf_X,W) + b
        else:           
            unscaled_log_prop = tf.matmul(tf_X,W)
        
        y_hat = tf.nn.softmax(logits = unscaled_log_prop + self.epsss)
        arg_max_class = tf.argmax(y_hat,1,output_type=tf.int32)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            if output_probs:
                res = y_hat.eval()
            else:
                res = arg_max_class.eval()
        
        return res
           
if __name__ == "__main__":
    
    from utils import bc_train_test_split, fmnist_train_test_split
    seed = 12123
    l2_pen = 0
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    X_train, y_train, X_test, y_test = bc_train_test_split(seed=seed,test_prop= 0.4)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logreg = LogReg(bias= True,iters=500,l2_penalty=l2_pen)
    logreg.fit(X_train,y_train)
    y_hat = logreg.predict(X_test)
    
    test_accuracy = np.mean(np.equal(y_hat,y_test))
   
    print("Test accuracy with bc: ", test_accuracy)
    
    X_train, y_train, X_test, y_test = fmnist_train_test_split()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logreg = LogReg(bias= True,iters=500,l2_penalty=l2_pen)
    logreg.fit(X_train,y_train)
    y_hat = logreg.predict(X_test)
    
    test_accuracy = np.mean(np.equal(y_hat,y_test))
   
    print("Test accuracy with fmnist: ", test_accuracy)