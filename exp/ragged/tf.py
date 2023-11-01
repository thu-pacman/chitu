import tensorflow as tf

rt = tf.ragged.constant(
    [
        [
            [3, 1, 4, 1],
            [3, 1, 4, 1],
        ],
        [
            [3, 2, 4, 1],
            [3, 1, 4, 1],
            [3, 1, 4, 1],
        ],
    ],
    ragged_rank=1
)

b = tf.constant([[1, 2, 1, 1], [3, 4, 1, 1], [3, 4, 1, 1], [3, 4, 1, 1]])

print(f'{rt.shape=}')
print(f'{rt.bounding_shape()=}')
print(f'{b.shape=}')
print(tf.add(rt, 3))
# print(rt@b)

print(rt.with_flat_values(tf.matmul(rt.flat_values, b)))
