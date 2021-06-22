import os
import numpy as np
import tensorflow as tf
from absl import flags, app
from libs.models import get_yolo
from libs.losses import get_losses
from libs.utils import transfer_weights
from libs.data_loaders.coco_loader import CocoDataLoader
from libs.data_loaders.prep_labels import PrepLabels
from configs import Configs, ProjectPath, cfg


FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', default=cfg.epochs, help='Number of training epochs')
flags.DEFINE_float('init_lr', default=cfg.init_lr, help='Initial learning rate')
flags.DEFINE_float('end_lr', default=cfg.end_lr, help='End learning rate')
flags.DEFINE_integer('warmup_epochs', default=cfg.warmup_epochs, help='Warm-up epochs')
flags.DEFINE_integer('batch_size', default=cfg.batch_size, help='Batch size')
flags.DEFINE_boolean('transfer_coco', default=True, help='Transfer pretrained coco weights')
flags.DEFINE_boolean('validation', default=True, help='Training with validation')


# Save some gpu errors
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)


def main(_argv):
    global yolo, optimizer
    global tb_train_writer, tb_val_writer
    global global_steps, warmup_steps, total_steps

    # Checkpoint Save Directory
    ckpt_save_dir = os.path.join(ProjectPath.CKPTS_DIR.value, cfg.project_name)
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    # Model
    yolo = get_yolo(ckpt_path=None, training=True, cfg=cfg)
    if FLAGS.transfer_coco:
        _transfer_coco()
    
    # Dataset & Tensorboard
    data_loader_options = {'project_name': cfg.project_name, 'batch_size': FLAGS.batch_size, 'input_size': cfg.input_size}
    train_ds = CocoDataLoader(stage='train', shuffle=True, aug=True, **data_loader_options)
    tb_train_writer = tf.summary.create_file_writer(ProjectPath.LOGS_DIR.value)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()
        
    # Epochs and Steps
    steps_per_epoch = len(train_ds.__iter__())
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = FLAGS.warmup_epochs * steps_per_epoch
    total_steps = FLAGS.epochs * steps_per_epoch

    # Training loss print format
    train_print_form = 'Epoch:{:2.0f}/{}, Step:{:5.0f}/{}, LR:{:.6f} | giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f} | Total Loss:{:7.2f}'

    if FLAGS.validation:
        # Validation Dataset
        val_ds = CocoDataLoader(stage='val', shuffle=False, aug=False, **data_loader_options)
        # Tensorboard
        tb_val_writer = tf.summary.create_file_writer(ProjectPath.LOGS_DIR.value)
        # Loss
        val_lowest_loss = 10000
        val_giou_loss = tf.keras.metrics.Mean(name='val_giou_loss')
        val_conf_loss = tf.keras.metrics.Mean(name='val_conf_loss')
        val_prob_loss = tf.keras.metrics.Mean(name='val_prob_loss')
        val_total_loss = tf.keras.metrics.Mean(name='val_total_loss')
        val_print_form = '\n\n[Validation] giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f} | Total Loss:{:7.2f}\n\n'

    # Training
    for epoch in range(1, FLAGS.epochs+1):
        for step, (batch_imgs, batch_labels) in enumerate(train_ds, 1):
            batch_labels_per_scale = PrepLabels(batch_labels=batch_labels, cfg=cfg).get_prep()
            results = _train_step(batch_imgs, batch_labels_per_scale)
            # results: optimizer.lr, giou_loss, conf_loss, prob_loss, total_loss
            print(train_print_form.format(epoch, cfg.epochs, step, steps_per_epoch, results[0], results[1], results[2], results[3], results[4]))

        if not FLAGS.validation:
            ckpt_save_path = os.path.join(ckpt_save_dir, f'{cfg.model_name}_{epoch:0>4}.h5')
            yolo.save_weights(ckpt_save_path, save_format='h5')
            continue
        
        # ====== ====== Validation ====== ======
        for batch_imgs, batch_labels in val_ds:
            batch_labels_per_scale = PrepLabels(batch_labels=batch_labels, cfg=cfg).get_prep()
            results = _validate(batch_imgs, batch_labels_per_scale)
            val_giou_loss.update_state(results[0])
            val_conf_loss.update_state(results[1])
            val_prob_loss.update_state(results[2])
            val_total_loss.update_state(results[3])

        with tb_val_writer.as_default():
            tf.summary.scalar("val_loss/total_val", val_total_loss.result(), step=epoch)
            tf.summary.scalar("val_loss/giou_val", val_giou_loss.result(), step=epoch)
            tf.summary.scalar("val_loss/conf_val", val_conf_loss.result(), step=epoch)
            tf.summary.scalar("val_loss/prob_val", val_prob_loss.result(), step=epoch)
        tb_val_writer.flush()
        print(val_print_form.format(val_giou_loss.result(), val_conf_loss.result(), val_prob_loss.result(), val_total_loss.result()))

        if val_total_loss.result() < val_lowest_loss:
            ckpt_save_path = os.path.join(ckpt_save_dir, f'{cfg.model_name}_{epoch:0>4}.h5')
            yolo.save_weights(ckpt_save_path, save_format='h5')
            val_lowest_loss = val_total_loss.result()

        val_giou_loss.reset_state()
        val_conf_loss.reset_state()
        val_prob_loss.reset_state()
        val_total_loss.reset_state()
        # ====== ====== ====== ====== ====== ======


def _train_step(batch_imgs, batch_labels_per_scale):
    with tf.GradientTape() as tape:
        yolo_outputs = yolo(batch_imgs, training=True)
        giou_loss, conf_loss, prob_loss = 0, 0, 0

        for i in range(cfg.num_scales):
            outputs_raw, outputs_decoded = yolo_outputs[i*2], yolo_outputs[i*2+1]
            label_bboxes, bboxes = batch_labels_per_scale[i]
            stride = cfg.strides[i]
            loss_items = get_losses(
                pred_raw=outputs_raw,
                pred_decoded=outputs_decoded,
                label=label_bboxes,
                bboxes=bboxes,
                stride=stride,
                iou_loss_thr=cfg.iou_loss_thr,
                num_classes=cfg.num_classes,
            )
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        gradients = tape.gradient(total_loss, yolo.trainable_variables)
        optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

    _warmup_lr()

    # writing summary data
    with tb_train_writer.as_default():
        tf.summary.scalar('lr', optimizer.lr, step=global_steps)
        tf.summary.scalar('loss/total_loss', total_loss, step=global_steps)
        tf.summary.scalar('loss/giou_loss', giou_loss, step=global_steps)
        tf.summary.scalar('loss/conf_loss', conf_loss, step=global_steps)
        tf.summary.scalar('loss/prob_loss', prob_loss, step=global_steps)
    tb_train_writer.flush()
    return optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


def _validate(batch_imgs, batch_labels_per_scale):
    yolo_outputs = yolo(batch_imgs, training=False)
    giou_loss, conf_loss, prob_loss = 0, 0, 0

    for i in range(cfg.num_scales):
        outputs_raw, outputs_decoded = yolo_outputs[i*2], yolo_outputs[i*2+1]
        label_bboxes, bboxes = batch_labels_per_scale[i]
        stride = cfg.strides[i]
        loss_items = get_losses(
            pred_raw=outputs_raw,
            pred_decoded=outputs_decoded,
            label=label_bboxes,
            bboxes=bboxes,
            stride=stride,
            iou_loss_thr=cfg.iou_loss_thr,
            num_classes=cfg.num_classes,
        )
        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
    total_loss = giou_loss + conf_loss + prob_loss
    return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


def _transfer_coco():
    coco_config = Configs()
    coco_config.num_classes = 80
    if cfg.model_name.endswith('_tiny'):
        coco_ckpt_path = ProjectPath.COCO_PRETRAINED_TINY_CKPT_PATH.value
    else:
        coco_ckpt_path = ProjectPath.COCO_PRETRAINED_CKPT_PATH.value
    pretrained_coco_yolo = get_yolo(ckpt_path=coco_ckpt_path, training=False, cfg=coco_config)
    transfer_weights(src_model=pretrained_coco_yolo, dst_model=yolo)


def _warmup_lr():
    # update learning rate
    # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
    global_steps.assign_add(1)
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * FLAGS.init_lr
    else:
        lr = FLAGS.end_lr + 0.5 * (FLAGS.init_lr - FLAGS.end_lr) * ((1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
    optimizer.lr.assign(lr.numpy())


if __name__ == '__main__':
    app.run(main)
