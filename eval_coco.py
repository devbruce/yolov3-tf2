import tensorflow as tf
from absl import flags, app
from libs.inference import YoloInf
from libs.evals.coco import GetCocoEval


FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', default=None, help='Checkpoint file path')
flags.DEFINE_string('img_prefix', default=None, help='Image directory path to evaluate', short_name='i')
flags.DEFINE_string('coco_gt', default=None, help='COCO GT file path', short_name='g')
flags.DEFINE_float('conf_thr', default=0.05, help='Inference confidence threshold')
flags.DEFINE_list('img_exts', default=['.png', '.jpg', '.jpeg'], help='Image extensions')
flags.mark_flag_as_required('ckpt')
flags.mark_flag_as_required('img_prefix')
flags.mark_flag_as_required('coco_gt')
flags.mark_flag_as_required('conf_thr')
flags.mark_flag_as_required('img_exts')


# Save some gpu memories
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(device=physical_device, enable=True)


def main(_argv):
    yolo_inf = YoloInf(ckpt_path=FLAGS.ckpt)
    coco_eval = GetCocoEval(
        img_prefix=FLAGS.img_prefix,
        coco_gt_path=FLAGS.coco_gt,
        yolo_inf=yolo_inf,
        conf_thr=FLAGS.conf_thr,
        img_exts=FLAGS.img_exts,
    )
    coco_eval.get(verbose=True)


if __name__ == '__main__':
    app.run(main)
