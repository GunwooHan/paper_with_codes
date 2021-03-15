import tensorflow as tf

class DownConvolution(tf.keras.layers.Layer):
    def __init__(self):
        super(DownConvolution,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=2,padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv3 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=2,padding='same')
        self.conv4 = tf.keras.layers.Conv2D(256,kernel_size=3,strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
                   
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.bn1(x , training = training)
        x = tf.nn.relu(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x, training= training)
        x = tf.nn.relu(x)
        return x 

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
               
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training= training)
        return x + input_tensor

class UpConvolution(tf.keras.layers.Layer):
    def __init__(self):
        super(UpConvolution,self).__init__()
        self.trans_conv1 = tf.keras.layers.Conv2DTranspose(128,kernel_size=3,strides=2,padding='same')
        self.trans_conv2 = tf.keras.layers.Conv2DTranspose(128,kernel_size=3,strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
    
        self.trans_conv3 = tf.keras.layers.Conv2DTranspose(64,kernel_size=3,strides=2,padding='same')
        self.trans_conv4 = tf.keras.layers.Conv2DTranspose(64,kernel_size=3,strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
    
    def call(self,input_tensor,training=False):
        x = self.trans_conv1(input_tensor)
        x = self.trans_conv2(x)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        return x

class generator(tf.keras.Model):
    def __init__(self):
        super(generator,self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64,kernel_size=7,strides=1,padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.down_conv = DownConvolution()
        
        self.residual1 = ResidualBlock()
        self.residual2 = ResidualBlock()
        self.residual3 = ResidualBlock()
        self.residual4 = ResidualBlock()
        self.residual5 = ResidualBlock()
        self.residual6 = ResidualBlock()
        self.residual7 = ResidualBlock()
        self.residual8 = ResidualBlock()
        
        self.up_conv = UpConvolution()
        
        self.conv2 = tf.keras.layers.Conv2D(3, kernel_size=7, strides=1,padding='same')
        
    def call(self,input_tensor,training=False):
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        
        x = self.down_conv(x)
        
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.residual6(x)
        x = self.residual7(x)
        x = self.residual8(x)
        
        x = self.up_conv(x)
        x = self.conv2(x)
        x = tf.nn.tanh(x)
        return x


class discriminator(tf.keras.Model):
    def __init__(self):
        super(discriminator,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32,kernel_size=3,strides=1,padding='same')
        
        self.conv2 = tf.keras.layers.Conv2D(64,kernel_size=3, strides=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv4 = tf.keras.layers.Conv2D(128,kernel_size=3,strides=2,padding='same')
        self.conv5 = tf.keras.layers.Conv2D(256,kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        
        self.conv6 = tf.keras.layers.Conv2D(256,kernel_size=3, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()      
        
        self.conv7 = tf.keras.layers.Conv2D(1,kernel_size=3, strides=1, padding='same')
    
    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = tf.nn.leaky_relu(x)
        
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn1(x, training= training)
        x = tf.nn.leaky_relu(x)
        
        x = self.conv4(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv5(x)
        x = self.bn2(x, training = training)
        x = tf.nn.leaky_relu(x)
        
        x = self.conv6(x)
        x = self.bn3(x, training= training)
        x = tf.nn.leaky_relu(x)
        
        x = self.conv7(x)
        return x    

if  __name__ == '__main__':
    # disc = discriminator()
    # gen = generator()
    # input_d = tf.keras.Input(shape=(None,None,3))
    # output_d = disc(input_d)
    # d = tf.keras.Model(inputs=input_d,outputs=output_d)

    # input_g = tf.keras.Input(shape=(None,None,3))
    # output_g = gen(input_g)
    # g = tf.keras.Model(inputs=input_g,outputs=output_g)

    d = discriminator()
    g = generator()
    d.build(input_shape=(None,None,None,3))
    g.build(input_shape=(None,None,None,3))
    # print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

    d.summary()
    g.summary()