import tensorflow as tf
def block(x, filters, multiplier):
    for _ in range(multiplier):
        y = conv2d(x,filters,1)
        y = conv2d(x,filters*2,3)
        x = y + x
    return x

def conv2d(x,filters, kernel_size, strides=(1,1)):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

input = tf.keras.layers.Input([416,416,3])
conv1 = conv2d(input,filters=32,kernel_size=3)
conv2 = conv2d(conv1,filters=64,kernel_size=3,strides=(2,2))
block1 = block(conv2,32,1)
conv3 = conv2d(block1,filters=128,kernel_size=3,strides=(2,2))
block2 = block(conv3,64,2)
conv4 = conv2d(block2,filters=256,kernel_size=3,strides=(2,2))
route1 = block(conv4,128,8)
conv5 = conv2d(route1,filters=512,kernel_size=3,strides=(2,2))
route2 = block(conv5,256,8)
conv6 = conv2d(route2,filters=1024,kernel_size=3,strides=(2,2))
block3 = block(conv6,512,4)


for _ in range(5):
    block3 = conv2d(block3,filters=1024,kernel_size=3)

route3 = block3

block3 = conv2d(block3,filters=1024,kernel_size=3)
out1  = tf.keras.layers.Conv2D(filters=255,kernel_size=3,strides=(1,1),padding='same',name='out1')(block3)



before1 = conv2d(route3,filters=512,kernel_size=3)
upsample1 = tf.keras.layers.UpSampling2D(size=(2,2),data_format="channels_last")(before1)

concat1 = tf.keras.layers.Concatenate(axis=-1)([upsample1, route2])

for _ in range(5):
    concat1 = conv2d(concat1,filters=512,kernel_size=3)

route4 = concat1

concat1 = conv2d(concat1,filters=512,kernel_size=3)
out2  = tf.keras.layers.Conv2D(filters=255,kernel_size=3,strides=(1,1),padding='same',name='out2')(concat1)


before2 = conv2d(route4,filters=256,kernel_size=3)
upsample2 = tf.keras.layers.UpSampling2D(size=(2,2),data_format="channels_last")(before2)

concat2 = tf.keras.layers.Concatenate(axis=-1)([upsample2, route1])

for _ in range(5):
    concat2 = conv2d(concat2,filters=256,kernel_size=3)


concat2 = conv2d(concat2,filters=256,kernel_size=3)
out3  = tf.keras.layers.Conv2D(filters=255,kernel_size=3,strides=(1,1),padding='same',name='out3')(concat2)


model = tf.keras.Model(input,[out1, out2, out3])
model.summary()
