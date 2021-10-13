import tensorflow as tf
import pandas as pd
import numpy as np
import util

def calRootMse(pre, label):
    pre = pre.reshape(-1)
    label = label.reshape(-1)
    return np.sqrt(np.mean((pre - label) ** 2))


def calPcrr(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    t = -103
    tp = len(y_true[(y_true < t) & (y_pred < t)])
    fp = len(y_true[(y_true >= t) & (y_pred < t)])
    fn = len(y_true[(y_true < t) & (y_pred >= t)])
    # noinspection PyBroadException
    try:
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        pcrr = 2 * (precision * recall) / (precision + recall)
    except:
        return 0
    return pcrr


def linear_layer(inputs, output_dims):
    x = tf.layers.dense(inputs, output_dims, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


def network(x):
    y1 = linear_layer(x, 2048)
    y1 = linear_layer(y1, 1024)
    y1 = linear_layer(y1, 256)

    y2 = tf.layers.dense(x, 256)
    y = tf.concat((y1, y2), axis=0)
    y = tf.layers.dense(y, 1)
    return y


train_data_np = np.array(train_data)
train_target_np = np.array(train_target).reshape(-1, 1)
eval_data_np = np.array(eval_data)
eval_target_np = np.array(eval_target).reshape(-1, 1)

train_size = len(train_target_np)
eval_size = len(eval_target_np)
input_dims = train_data_np.shape[-1]

batch_size = 1280
epoch_num = 10000

train_size = train_size // batch_size * batch_size
eval_size = eval_size // batch_size * batch_size
train_data_np = train_data_np[: train_size]
train_target_np = train_target_np[: train_size]
eval_data_np = eval_data_np[: eval_size]
eval_target_np = eval_target_np[: eval_size]
print('train_size', train_size, 'eval_size', eval_size)
input_pl = tf.placeholder(tf.float32, (None, input_dims))
label_pl = tf.placeholder(tf.float32, (None, 1))
outputs = network(input_pl)
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 2e-5
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.8, staircase=True)
loss = tf.losses.mean_squared_error(labels=label_pl, predictions=outputs)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8)
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(init)
best_epoch_idx, best_loss = 0, 1e10
train_losses, eval_losses = [], []

for epoch_index in range(epoch_num):
    print('Runing')
    print('lf', sess.run(learning_rate, {global_step: epoch_index}))
    total_loss = 0
    for i in range(0, train_size, batch_size):
        datas, labels = train_data_np[i: i+batch_size, ...], train_target_np[i: i+batch_size, ...]
        los, _, out = sess.run([loss, train_op, outputs], feed_dict={input_pl: datas, label_pl: labels})
        total_loss += los / (train_size / batch_size)
    print('train epoch {}˖ {}'.format(epoch_index, total_loss))
    train_losses.append(total_loss)
    total_loss = 0
    result = []
    for i in range(0, eval_size, batch_size):
        datas, labels = eval_data_np[i: i+batch_size, ...], eval_target_np[i: i+batch_size, ...]
        los, out = sess.run([loss, outputs], feed_dict={input_pl: datas, label_pl: labels})
        total_loss += los / (eval_size / batch_size)
        result.append(np.array(out))
    result = np.concatenate(result, axis=0)
    print('eval epoch {}: {}, rmse: {}, pcrr : {}'.format(epoch_index, total_loss, calRootMse(result, eval_target_np), calPcrr(result, eval_target_np)))
    eval_losses.append(total_loss)

    if total_loss <= best_loss:
        best_loss, best_epoch_idx = total_loss, epoch_index
        tf.saved_model.simple_save(sess, model_save_path+'{}'.format(best_loss),
        inputs={"myInput": input_pl}, outputs={"myOutput": outputs})
    if epoch_index - best_epoch_idx >= 10:
        break