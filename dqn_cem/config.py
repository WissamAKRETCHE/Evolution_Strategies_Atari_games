import tensorflow as tf
import os

tf.flags.DEFINE_string("mode", "train", "choices=[train, test]")
tf.flags.DEFINE_string("model", "dqn_atari", "NN model")
tf.flags.DEFINE_string("memory", "sequential", "memory type")
tf.flags.DEFINE_string("agent", "dqn", "for example, dqn")
tf.flags.DEFINE_string("game", "Pong-v0", "game environment. Ex: Humanoid-v1, OffRoadNav-v0")
tf.flags.DEFINE_string("exp_dir", ".", "game environment. Ex: Humanoid-v1, OffRoadNav-v0")
tf.flags.DEFINE_string("base_dir", ".", "Directory to write summaries and models to.")
tf.flags.DEFINE_string("save_path", ".", "Directory to write summaries and models to.")
tf.flags.DEFINE_integer("save_weight_interval", 10000, "number of steps before saving the model weight")
tf.flags.DEFINE_integer("save_log_interval", 100, "number of episodes before saving the model weight")
tf.flags.DEFINE_integer("max_steps", 5000, "max number of steps for the whole training")
tf.flags.DEFINE_integer("num_actions", 5000, "max number of steps for the whole training")
tf.flags.DEFINE_integer("memory_limit", 1000000, "memory size")
tf.flags.DEFINE_integer("memory_window_length", 4, "memory window size")
tf.flags.DEFINE_boolean("visualize_train", True, "visualize the training process")


def parse_flags():
    FLAGS = tf.flags.FLAGS
    # FLAGS.exp_dir = os.getcwd() + "/{}/{}/{}/".format(FLAGS.base_dir, FLAGS.game, FLAGS.agent)

    # Keras use relative path to save models
    FLAGS.exp_dir = "{}/{}/{}/".format(".", FLAGS.game, FLAGS.agent)

    FLAGS.log_dir = FLAGS.exp_dir + "/log"
    FLAGS.save_path = FLAGS.exp_dir + "/model"
    return FLAGS
