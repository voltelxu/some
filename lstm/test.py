import tensorflow as tf
import helper
import model

def train():
    '''
    config param
    '''

    batch_size = 64
    embed_size = 128
    voc_size = 1613
    hidden_size = 400
    num_layers = 1
    decay_steps = 12000
    decay_rate = 0.9
    l2_lambda = 0.0001
    learn_rate = 0.01
    is_train = True
    lable_size = 12
    prob = 0.7
    num_epoch = 10
    save_path='./path/'

    train = helper.data()
    train.getdata('train')
    train_len = len(train.data)
    train_size = train_len/batch_size
    
    test = helper.data()
    test.getdata('test')
    test_len = len(test.data)
    test_size = test_len/batch_size

    pr = open('out', 'a')

    with tf.Session() as sess:
        m = model.model(learn_rate=learn_rate, batch_size=batch_size, embed_size=embed_size, voc_size=voc_size, classes=lable_size, hidden_size=hidden_size, num_layers=num_layers, decay_steps=decay_steps, decay_rate=decay_rate, l2_lambda=l2_lambda, is_train=is_train)
        sess.run(tf.global_variables_initializer())
        sess.run(m.epoch_step)

        saver = tf.train.Saver()
        for epoch in range(num_epoch):
            la = 0.0
            ac = 0.0
            train.index = 0
            for j in range(train_size):
                x, y, l = train.getbatch(batch_size)
                loss, acc, _ = sess.run([m.loss, m.accuracy, m.train_op], feed_dict={m.input_data:x, m.target:y, m.seqlen:l,m.prob:prob})
                la = la + loss
                ac = ac + acc
            ac = ac / float(train_size)
            a = 'train accuracy : ' + str(ac)+ ' ' + 'train loss : ' + str(la)
            pr.write(a + '\n')
            print a 
            ac = 0.0
            la = 0.0
            sess.run(m.epoch_increment)
            test.index = 0
            for k in range(test_size):
                x, y, l = test.getbatch(batch_size)
                loss, acc = sess.run([m.loss, m.accuracy], feed_dict={m.input_data:x, m.target:y, m.seqlen:l,m.prob:1.0})
                ac = ac + acc
                la = la + loss
            ac = ac / float(test_size)
            a = 'test accuracy : ' + str(ac) + ' ' + 'test loss : ' + str(la)
            pr.write(a +'\n')
            print a
            saver.save(sess, save_path, global_step=epoch)
    pr.close()

train()
