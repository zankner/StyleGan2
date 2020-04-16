import tensorflow as tf


def wasserstein_loss(real_scores, fake_scores):
    batch_size = real_scores.shape[0]

    avg_real_scores = tf.math.reduce_mean(real_scores)
    avg_fake_scores = tf.math.reduce_mean(avg_fake_scores)

    gen_loss = -avg_fake_score

    alpha = tf.random.uniform([batch_size, 1, 1, 1])
    interpolated = (alpha * generated) + ((1 - alpha) * real)
    critic_interpolated = discriminator(interpolated)
    critic_gradient = tf.gradient(critic_interpolated, interpolated)
    norm_critic_gradient = tf.math.sqrt(
        tf.reduce_sum(tf.math.square(critic_gradient), [1, 2, 3]))
    norm_critic_center = norm_critic_gradient - 1
    gradient_penalty = tf.math.square(norm_critic_center)

    discrim_loss = -avg_real_scores + avg_fake_scores + (gp_weight *
                                                         gradient_penalty)

    return gen_loss, discrim_loss
