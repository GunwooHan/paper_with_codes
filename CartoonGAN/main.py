import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time 
import os
from tqdm import tqdm
import argparse
import multiprocessing
import parmap

from models import generator, discriminator
from utils import make_edge_smooth, src_load, tgt_load
from losses import adversarial_loss, generator_loss, content_loss

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_epoch', type=int, default=10, help='pretrain step for generator only using content loss while training')
    parser.add_argument('--epoch', type=int, default=200, help='train epoch')
    parser.add_argument('--batch_size',type=int, default=4, help = 'batch_size')
    parser.add_argument('--src_dir',type=str, default='./src_data', help = 'directory of source image data')
    parser.add_argument('--tgt_dir',type=str, default='./target_data', help = 'directory of target image data')

    args = parser.parse_args()

    if os.path.isdir('smooth_data'):
        pass
    else:
        os.mkdir('smooth_data')
        filenames = os.listdir(f'{args.tgt_dir}')
        print('smooth_target_data generating ...')
        for filename in tqdm(filenames):
            img = cv2.imread(f'target_data/{filename}')
            img = make_edge_smooth(img)
            cv2.imwrite(f"smooth_data/{filename}",img)


    src_ds = tf.data.Dataset.list_files(f'{args.src_dir}/*', shuffle=False)
    orig_tgt_ds = tf.data.Dataset.list_files(f'{args.src_dir}/*', shuffle=False)
    sms_ds = tf.data.Dataset.list_files('./smooth_data/*', shuffle=False)

    src_ds = src_ds.map(src_load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    orig_tgt_ds = orig_tgt_ds.map(tgt_load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    sms_ds = sms_ds.map(tgt_load, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    src_ds = src_ds.map(lambda x: x/127.5-1)
    orig_tgt_ds = orig_tgt_ds.map(lambda x: x/127.5-1)
    sms_ds = sms_ds.map(lambda x: x/127.5-1)

    tgt_ds = tf.data.Dataset.zip((orig_tgt_ds,sms_ds))

    src_ds = src_ds.shuffle(400).batch(args.batch_size)
    tgt_ds = tgt_ds.shuffle(400).batch(args.batch_size)

    print(len(src_ds))
    print(len(tgt_ds))

    disc = discriminator()
    gen = generator()
    input_d = tf.keras.Input(shape=(None,None,3))
    output_d = disc(input_d)
    d = tf.keras.Model(inputs=input_d,outputs=output_d)

    input_g = tf.keras.Input(shape=(None,None,3))
    output_g = gen(input_g)
    g = tf.keras.Model(inputs=input_g,outputs=output_g)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    vgg19 = tf.keras.applications.VGG19(weights='imagenet',input_shape=(None,None,3),include_top=False)
    feature_extracter = tf.keras.Model(inputs=vgg19.input,outputs=vgg19.get_layer('block4_conv4').output)

    def VGG(input_tensor,feature_extracter):
        return feature_extracter(input_tensor)

    for epoch in range(1,args.pretrain_epoch+1):
        pretrain_loss = 0.0
        start_time = time.time()
        for src_batch in tqdm(src_ds):
            with tf.GradientTape() as gen_tape:
                generated_img = g(src_batch, training=True)

                reconstruct_feature = VGG(generated_img,feature_extracter) 
                src_feature = VGG(src_batch,feature_extracter)

                c_loss = 10 * content_loss(reconstruct_feature,src_feature)
                pretrain_loss += c_loss

            gradients_of_generator = gen_tape.gradient(c_loss, g.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))

        print(f'epoch : {epoch}, time : {time.time()-start_time} , pretrain_loss : {pretrain_loss}')


    tf.keras.models.save_model(g,'pretrain')

    if os.path.isdir('results'):
        pass
    else:
        os.mkdir('results')

    for epoch in range(1,args.epoch+1):
        start_time = time.time()
        total_loss=0.0
        total_gen_loss = 0.0
        total_adv_loss = 0.0
        for src_batch, tgt_batch in tqdm(zip(src_ds,tgt_ds)):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_img = g(src_batch, training=True)

                reconstruct_feature = VGG(generated_img,feature_extracter) 
                src_feature = VGG(src_batch,feature_extracter)

                c_loss = content_loss(reconstruct_feature,src_feature)

                fake_loss = d(generated_img,training=True)
                real_loss = d(tgt_batch[0],training=True)
                smooth_loss = d(tgt_batch[1],training=True)

                adv_loss = adversarial_loss(real_loss, smooth_loss, fake_loss)
                gen_loss = generator_loss(fake_loss)


                total_gen_loss += gen_loss + 10 * c_loss
                total_adv_loss += adv_loss
                
                total_loss += adv_loss + total_gen_loss

            gradients_of_generator = gen_tape.gradient(total_gen_loss, g.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(adv_loss, d.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
        total_loss = total_loss/len(src_batch)
        total_gen_loss = total_gen_loss/len(src_batch)
        total_adv_loss = total_adv_loss/len(src_batch)
        
        tf.keras.models.save_model(g,f'checkpoint/generator_{epoch}')
        tf.keras.models.save_model(d,f'checkpoint/discriminator_{epoch}')
        
        if os.path.isdir(f'results/{epoch}'):
            pass
        else:
            os.mkdir(f'results/{epoch}')

        filenames = os.listdir('src_data')
        for filename in tqdm(filenames):
            img_array = np.fromfile(f'src_data/{filename}', np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = g((img[tf.newaxis,...]/127.5-1).astype('float32'))
            cv2.imwrite(f"results/{epoch}/{filename}",(((img[0]+1)*127.5).numpy()).astype(np.uint8))

        print(f'epoch : {epoch}, time : {time.time()-start_time} , total_loss : {total_loss} , gen_loss : {total_gen_loss} ,disc_loss : {adv_loss}')