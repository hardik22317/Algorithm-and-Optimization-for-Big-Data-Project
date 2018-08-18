
# coding: utf-8

# # Teaching a Deep Convolutional Generative Adversarial Network (DCGAN) to draw MNIST characters
# 
# In the last tutorial, we learnt using tensorflow for designing a Variational Autoencoder (VAE) that could draw MNIST characters. Most of the created digits looked nice. There was only one drawback -- some of the created images looked a bit fuzzy. The VAE was trained with the _mean squared error_ loss function. However, it's quite difficult to encode exact character edge locations, which leads to the network being unsure about those edges. And does it really matter if the edge of a character starts a few pixels more to the left or right? Not really.
# In this article, we will see how we can train a network that does not depend on the mean squared error or any related loss functions--instead, it will learn all by itself what a real image should look like.
# 
# ## Deep Convolutional Generative Adversarial Networks
# Another network architecture for learning to generate new content is the DCGAN. Like the VAE, our DCGAN consists of two parts:
# * The _discriminator_ learns how to distinguish fake from real objects of the type we'd like to create
# * The _generator_ creates new content and tries to fool the discriminator
# 
# There is a HackerNoon article by Chanchana Sornsoontorn that explains the concept in more detail and describes some creative projects DCGANs have been applied to. One of these projects is the generation of MNIST characters. Let's try to use python and tensorflow for the same purpose.

# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[ ]:
tf.reset_default_graph()

def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


# ## Setting up the basics
# Like in the last tutorial, we use tensorflow's own method for accessing batches of MNIST characters. We set our batch size to be 64. Our generator will take noise as input. The number of these inputs is being set to 100. Batch normalization considerably improved the training of this network. For tensorflow to apply batch normalization, we need to let it know whether we are in training mode. The variable _keep_prob_ will be used by our dropout layers, which we introduce for more stable learning outcomes.
# _lrelu_ defines the popular leaky ReLU, that hopefully will be supported by future versions of tensorflow! I firstly tried to apply standard ReLUs to this network, but this lead to the well-known _dying ReLU problem_, and I received generated images that looked like artwork by Kazimir Malevich--I just got black squares. 
# 
# Then, we define a function _binary_cross_entropy_, which we will use later, when computing losses.

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')
tf.reset_default_graph()
batch_size = 64
n_noise = 64

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))

def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# ## The discriminator
# Now, we can define the discriminator. It looks similar to the encoder part of our VAE. As input, it takes real or fake MNIST digits (28 x 28 pixel grayscale images) and applies a series of convolutions. Finally, we use a sigmoid to make sure our output can be interpreted as the probability to that the input image is a real MNIST character.

# In[1]:


def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x


# ## The generator
# The generator--just like the decoder part in our VAE--takes noise and tries to learn how to transform this noise into digits. To this end, it applies several transpose convolutions. At first, I didn't apply batch normalization to the generator, and its learning seemed to be really unefficient. After applying batch normalization layers, learning improved considerably. Also, I firstly had a much larger dense layer accepting the generator input. This led to the generator creating the same output always, no matter what the input noise was. Tuning the generator honestly took quite some effort!

# In[2]:


def generator(z, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x    


# ## Loss functions and optimizers
# Now, we wire both parts together, like we did for the encoder and the decoder of our VAE in the last tutorial.
# However, we have to create two objects of our discriminator
# * The first object receives the real images
# * The second object receives the fake images
# 
# _reuse_ of the second object is set to _True_ so both objects share their variables. We need both instances for computing two types of losses:
# * when receiving real images, the discriminator should learn to compute high values (near _1_), meaning that it is confident the input images are real
# * when receiving fake images, it should compute low values (near _0_), meaning it is confident the input images are not real
# 
# To accomplish this, we use the _binary cross entropy_ function defined earlier. The generator tries to achieve the opposite goal, it tries to make the discriminator assign high values to fake images.
# 
# Now, we also apply some regularization. We create two distinct optimizers, one for the discriminator, one for the generator. We have to define which variables we allow these optimizers to modify, otherwise the generator's optimizer could just mess up the discriminator's variables and vice-versa.
# 
# We have to provide the __update_ops__ to our optimizers when applying batch normalization--take a look at the tensorflow documentation for more information on this topic.

# In[3]:

w=3;


# g_arr = tf.TensorArray(size=w,dtype=tf.variant);
# d_real_arr = tf.TensorArray(size=w,dtype=tf.variant);
# d_fake_arr = tf.TensorArray(size=w,dtype=tf.variant);

# vars_g_arr = tf.TensorArray(size=w,dtype=tf.variant);
# vars_d_arr = tf.TensorArray(size=w,dtype=tf.variant);

# d_reg_arr = tf.TensorArray(size=w,dtype=tf.variant);
# g_reg_arr = tf.TensorArray(size=w,dtype=tf.variant);

# loss_d_real_arr = tf.TensorArray(size=w,dtype=tf.variant);
# loss_d_fake_arr = tf.TensorArray(size=w,dtype=tf.variant);
# loss_g_arr = tf.TensorArray(size=w,dtype=tf.variant);
# loss_d_arr = tf.TensorArray(size=w,dtype=tf.variant);
# noise_arr=tf.TensorArray(size=w,dtype=tf.variant);
# keep_prob_arr=tf.TensorArray(size=w,dtype=tf.variant);
# is_training_arr=tf.TensorArray(size=w,dtype=tf.variant);

g_arr =[]
d_real_arr = []
d_fake_arr = []

vars_g_arr = []
vars_d_arr = []

d_reg_arr = []
g_reg_arr = []

loss_d_real_arr = []
loss_d_fake_arr = []
loss_g_arr = []
loss_d_arr = []
noise_arr=[]
keep_prob_arr=[]
is_training_arr=[]

for i in range(w):
    print(i,"i")
    g_arr.append(generator(noise, keep_prob, is_training))
    d_real_arr.append(discriminator(X_in))
    d_fake_arr.append(discriminator(g_arr[i], reuse=True))

    vars_g_arr.append([var for var in tf.trainable_variables() if var.name.startswith("generator")])
    vars_d_arr.append([var for var in tf.trainable_variables() if var.name.startswith("discriminator")])

    d_reg_arr.append(tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d_arr[i]))
    g_reg_arr.append(tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g_arr[i]))

    loss_d_real_arr.append( binary_cross_entropy(tf.ones_like(d_real_arr[i]), d_real_arr[i]))
    loss_d_fake_arr.append( binary_cross_entropy(tf.zeros_like(d_fake_arr[i]), d_fake_arr[i]))
    loss_g_arr.append( tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake_arr[i]), d_fake_arr[i])))
    loss_d_arr.append( tf.reduce_mean(0.5 * (loss_d_real_arr[i] + loss_d_fake_arr[i])))
    
    # g_arr = g_arr.write(i,generator(noise, keep_prob, is_training))
    # d_real_arr = d_real_arr.write(i,discriminator(X_in))
    # d_fake_arr = d_fake_arr.write(i,discriminator(g_arr.read(i), reuse=True))

    # vars_g_arr = vars_d_arr.write(i,[var for var in tf.trainable_variables() if var.name.startswith("generator")])
    # vars_d_arr = vars_d_arr.write(i,[var for var in tf.trainable_variables() if var.name.startswith("discriminator")])

    # d_reg_arr = d_reg_arr.write(i,tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d_arr.read(i)))
    # g_reg_arr = g_reg_arr.write(i,tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g_arr.read(i)))

    # loss_d_real_arr = loss_d_real_arr.write(i, binary_cross_entropy(tf.ones_like(d_real_arr.read(i)), d_real_arr.read(i)))
    # loss_d_fake_arr = loss_d_fake_arr.write(i, binary_cross_entropy(tf.zeros_like(d_fake_arr.read(i)), d_fake_arr.read(i)))
    # loss_g_arr = loss_g_arr.write(i, tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake_arr.read(i)), d_fake_arr.read(i))))
    # loss_d_arr = loss_d_arr.write(i, tf.reduce_mean(0.5 * (loss_d_real_arr.read(i) + loss_d_fake_arr.read(i))))

# loss_d=loss_d_arr[0];
# for x in range(1,w):
#     loss_d=loss_d+loss_d_arr[x]
loss_d=loss_d_arr[w-1]

# d_reg=d_reg_arr[0];
# for x in range(1,w):
#     d_reg=d_reg+d_reg_arr[x]
d_reg=d_reg_arr[w-1]

loss_g=loss_g_arr[0];
for x in range(1,w):
    loss_g=loss_g+loss_g_arr[x]
loss_g=loss_g/w


g_reg=g_reg_arr[0];
for x in range(1,w):
    g_reg=g_reg+g_reg_arr[x]
g_reg=g_reg/w


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.0015).minimize(loss_d + d_reg, var_list=vars_d_arr[w-1])
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.0015).minimize(loss_g + g_reg, var_list=vars_g_arr[w-1])
    
y = tf.Variable(0, dtype=tf.int32,trainable=False,name='y')
iy = tf.assign_add(y,50,name = 'iy')

# y = tf.get_variable("y",shape=[0])
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#saver.restore(sess, "./output/model.ckpt")

# print("saver: ",saver)
x=sess.run(y);
print("yy: ",x)
        

# ## Training the DCGAN
# Finally, the fun part begins--let's train our network! 
# We feed random values to our generator, which will learn to create digits out of this noise. We also take care that neither the generator nor the discriminator becomes too strong--otherwise, this would inhibit the learning of the other part and could even stop the network from learning anything at all (I unfortunately have made this experience).

# In[4]:
losses=[]
for i in range(2500):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)   
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]  

    l=i;
    r=i-w+1
    if (r<0):
        r=0

    d_real_ls=[0]*64
    d_fake_ls=[0]*64
    g_ls=0
    d_ls=0

    for x in range(l,r,-1):
        d_real_ls_curr, d_fake_ls_curr, g_ls_curr, d_ls_curr = sess.run([loss_d_real_arr[x%w], loss_d_fake_arr[x%w], loss_g_arr[x%w], loss_d_arr[x%w]], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})
        # print(d_real_ls_curr.shape)
        d_real_ls = [d_real_ls[i]+d_real_ls_curr[i] for i in range(64)]

        # d_fake_ls = d_fake_ls + d_fake_ls_curr
        d_fake_ls = [d_fake_ls[i]+d_fake_ls_curr[i] for i in range(64)]
        g_ls = g_ls + g_ls_curr
        d_ls = d_ls + d_ls_curr
        
    # d_real_ls=d_real_ls/w
    d_real_ls[:] = [x / w for x in d_real_ls]
    # d_fake_ls=d_fake_ls/w
    d_fake_ls[:] = [x / w for x in d_fake_ls]
    g_ls=g_ls/w
    d_ls=d_ls/w

    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    
    # if g_ls * 1.5 < d_ls:
    #     train_g = False
    #     pass
    # if d_ls * 2 < g_ls:
    #     train_d = False
    #     pass
    
    # if train_d:
    sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
               
    # if train_g:
    sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})
        
    if not i % 50:
        print (i, d_ls, g_ls, d_real_ls, d_fake_ls)
        losses.append((d_ls,g_ls))       
        # if not train_g:
            # print("not training generator")
        # if not train_d:
            # print("not training discriminator")
        gen_img = sess.run(g_arr[i%w], feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        imgs = [img[:,:,0] for img in gen_img]
        m = montage(imgs)
        gen_img = m
        plt.axis('off')
        plt.imshow(gen_img, cmap='gray')
        x=sess.run(y);
        print("y: ",x)
        plt.savefig('./Images/epoch_'+ str(x)+'.png' )
        save_path = saver.save(sess, "./output/model.ckpt")
        #print("Model saved in path: %s" % save_path)
        #plt.show()
        # y=tf.add(y,50)
        sess.run(iy)


fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()

plt.show()

# ## Results
# Take a look at the pictures drawn by our generator--they look more realistic than the pictures drawn by the VAE, which looked more fuzzy at their edges. Training however took much longer than training the other model.
# 
# In conclusion, training the DCGAN took me much longer than training the VAE. Maybe fine-tuning the architecture could speed up the network's learning. Nonetheless, it's a real advantage that we are not dependent on loss functions based on pixel positions, making the results look less fuzzy. This is especially important when creating more complex data--e.g. pictures of human faces. So, just be a little patient--then everything is possible in the world of deep learning!
