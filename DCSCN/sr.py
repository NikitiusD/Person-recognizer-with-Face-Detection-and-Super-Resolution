import tensorflow as tf
import DCSCN
from helper import args

try:
    args.flags.DEFINE_string("file", "image.jpg", "Target filename")
except:
    pass
FLAGS = args.get()


def main(_):
    model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    model.init_all_variables()
    model.load_model()

    model.do_for_file(FLAGS.file, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
