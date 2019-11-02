# example of training the discriminator model on real and random cifar10 images
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy import where
from numpy.random import rand
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from matplotlib import pyplot
<<<<<<< HEAD
import NamedEntityMatching as NER
=======
>>>>>>> 14ecd7a7094fda4cd9eb1bec8b25b80af54c9657


#link to the lecture where this code is from
#https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/



# example of loading the cifar10 dataset
#Keras will automatically download a compressed version of the images and save them under your home directory in ~/.keras/datasets/
from keras.datasets.cifar10 import load_data

'''
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
'''

objectnameDict = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}


#tutorial for centering and normalizing images
#https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
# load and prepare cifar10 training images
#We are scaling the images in the pixel range [-1,1] because our generator model will be using
#the tanh activation function, so the pixel range will be [-1,1] for the fake images.
def load_real_samples(object):
    # load cifar10 dataset
    (trainX, trainY), (_, _) = load_data()
    # convert from unsigned ints to floats
    X = []
    for i in range(len(trainY)):
        if trainY[i][0]==objectnameDict[object]:
            X.append(trainX[i])
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


#We will use some real images from the CIFAR-10 dataset and some fake images to train our
#discriminative model. We will use random sampling to choose images for stochastic gradient 
#descent. We label them with '1'.
# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# use the generator to generate n fake examples, with class labels
#This is our real generator function. It takes as input n_samples number of points
#generated from a gaussian distribution. It is upto the generator to find the right 
#distribution from the latent space. The generator will try to update it's weights 
#in such a way so it can generate images that are assigned a probablity closer to 1 
#by the discriminator.
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


'''
# generate n fake samples with class labels
# We dont have a functioning generator model yet, but the actual generator model
# will generate images using tanh activation function so the pixel range will be [-1,1]
# and the label of all fake images will be '0'. We are making this dummy generator
# function that works like that.
def generate_fake_samples(n_samples):
    # generate uniform random numbers in [0,1]
    X = rand(32 * 32 * 3 * n_samples)
    # update to have the range [-1, 1]
    X = -1 + X * 2
    # reshape into a batch of color images
    X = X.reshape((n_samples, 32, 32, 3))
    # generate 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y
'''


# train the discriminator model
def train_discriminator(model, dataset, n_iter=20, n_batch=128):
#What is loss and what is accuracy in a model: https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_iter):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator on real samples
        real_loss, real_acc = model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(half_batch)
        # update discriminator on fake samples
        fake_loss, fake_acc = model.train_on_batch(X_fake, y_fake)
        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))


# # plot images from the training dataset
# for i in range(49):
# 	# define subplot
# #subplots are used to create multiple plots in one plots
# 	pyplot.subplot(7, 7, 1 + i)
# 	# turn off axis
# 	pyplot.axis('off')
# 	# plot raw pixel data
# #imshow finishes drawing a picture instead of painting it and show prints it
# 	pyplot.imshow(trainX[i])
# pyplot.show()


# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
    model = Sequential()
    # normal
#tutorial to understand convolutional layers: https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
#tutorial for pooling layers: https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
#tutorial for adam: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#we need a certain filter to extract simple features like detecting horizontal or vertical 
#lines. We use more filters to extract out complex patterns and shapes from the image.

#we use down sampling i.e. padding, strides etc to reduce the location dependency 
#of features. Because we just want to detect the presence of these features regardless 
#of where we find them.

#LeakyRelU instead of RelU to avoid the dying RelU problem where once the loss is negative
#the gradient is zero so the system can never update itself.

#adam takes into account not only what the gradient is but also how fast or slow the 
#gradient is changing. 
#TODO
#need to read about optimizers and adam more
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
#deconvolution well explained: https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
#We use kernel size that is a multiple of stride to avoid deconvolution checkerboard
#problem. Explained here: https://distill.pub/2016/deconv-checkerboard/
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# define the combined generator and discriminator model, for updating the generator
#In our composite gan model we are stacking our generator model and discriminator model 
#together. Our generator will generate fake images and feed it to the discriminator model.
#The output of the 1st layer which is the generator model will be a 32*32 image with 3 color channels.
#The will be the input to the discriminator model. The discriminator will output a binary classification.
#So the output of the whole composite model will be a binary classification. The model will try to
#minimize its loss based on that output. Basically it will try to update its weights in such
# a way so the label of the generated images generated from the first layer is assigned a 
#probablity closer to 1 by the second layer which is the output of the composite layer.
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# train the generator and discriminator
#Here we train discriminator twice per epoch, separately with fake and real samples.
#Then we generate latent points and feed that as input to the generator. This time we label
#the fake images with '1' even though they are fake. This is important because the discriminator
#will assign a lower probablity that is close to zero to these images. The generator
#as a result will have a higher loss, meaning the difference between 1 and the probablity assigned
#by the discriminator. So it will continuously try to generate images in a way so the images
#are assigned a higher probablity by the discriminator.
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            # save the generator model tile file
            filename = 'generator_model_%03d.h5' % (epoch+1)
            g_model.save(filename)
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))


#To evaluate a performance of gan, unfortunately we need a human operator because we 
#cannot evaluate gan models objectively. So we have three ways to do this.
#1. After certain number of epochs, calculate the disciminator accuracy and print it.
#2. After every certain number of epochs, save the generator model.
#3. Save the images generated by the corresponding generator model that we saved.
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))


# create and save a plot of generated images
#Function for saving the images generated by the generator model we will save periodically.
def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot
    save_plot(x_fake, epoch)
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


'''
# generate samples
#Generate fake samples using the untrained generator model and plot them.
model = define_generator(latent_dim)
n_samples = 49
X, _ = generate_fake_samples(model, latent_dim, n_samples)
# scale pixel values from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the generated samples
for i in range(n_samples):
    # define subplot
    pyplot.subplot(7, 7, 1 + i)
    # turn off axis labels
    pyplot.axis('off')
    # plot single image
    pyplot.imshow(X[i])
# show the figure
pyplot.show()
'''


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# summarize gan model
gan_model.summary()
# plot gan model
# plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
# load image data
#This is a dummy example to demonstrate how the named entity recognizer will communicate with
#image generator. This is very rudimentary and will be updated later.
#TODO: thorough readup of the spacy library
#https://spacy.io/
objects = NER.extract_objects()
objects[0] = 'frog'
dataset = load_real_samples(objects[0])
# train model
# train(g_model, d_model, gan_model, dataset, latent_dim)
