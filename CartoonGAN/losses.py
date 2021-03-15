import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
L1_loss = tf.keras.losses.MeanAbsoluteError()

def adversarial_loss(real_output, smooth_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    smooth_loss = cross_entropy(tf.ones_like(smooth_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    
    total_loss = real_loss + fake_loss + smooth_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def content_loss(src_feature, reconstruct_feature):
    return L1_loss(src_feature, reconstruct_feature)
