import dirt
import numpy as np
import tensorflow as tf

from dirt import matrices
from tensorflow.python.framework import ops


def perspective_projection(cam,f, c, w, h, near=0.1, far=10., name=None):
    """Constructs a perspective projection matrix.
    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.

    Returns:
        a 4x4 `Tensor` containing the projection matrix
    """

    with ops.name_scope(name, 'PerspectiveProjection', [f, c, w, h, near, far]) as scope:


        sx = cam[0]
        sy = cam[1]
        tx = cam[2]
        ty = cam[3]
        elements = [
            [sx, 0., 0., sx * tx],
            [0., sy, 0., -sy * ty],
            [0., 0., -1, 0.],
            [0., 0., 0., 1.]
        ]

        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))


def render_colored_batch(m_v, m_f, m_vc, width, height,cam, camera_f, camera_c, bgcolor=np.zeros(3, dtype=np.float32),
                         num_channels=3, camera_t=np.zeros(3, dtype=np.float32),
                         camera_rt=np.zeros(3, dtype=np.float32), name=None):
    # assert (num_channels == m_vc.shape[-1] == bgcolor.shape[0])



    view_matrix = matrices.compose(
        matrices.rodrigues(camera_rt.astype(np.float32)),
        matrices.translation(camera_t.astype(np.float32)),
    )
    vertices_split, faces_split = dirt.lighting.split_vertices_by_face(m_v, m_f)
    vertices_split = tf.concat([vertices_split, tf.ones_like(vertices_split[:, :, -1:])], axis=2)
    vertices_view_split = tf.einsum('evi,ij->evj', vertices_split, view_matrix)

    # projection_matrix = perspective_projection(cam,camera_f, camera_c, width, height, .1, 10)
    # projected_vertices_split = tf.einsum('eij,jk->eik', vertices_view_split, projection_matrix)
    vertex_normals_split = dirt.lighting.vertex_normals_pre_split(vertices_split, faces_split)

    projection_matrix = []
    for i in range(tf.shape(m_v)[0]):
        cam_tem = cam[i]
        projection_matrix.append(
            tf.expand_dims(
                perspective_projection(cam_tem, camera_f, camera_c, width, height, .1, 10),
                0
            )
        )

    projection_matrix = tf.concat(projection_matrix, 0)
    vertices_view_split = tf.einsum('eij,ejk->eik',vertices_view_split , projection_matrix)

    normals = dirt.rasterise_batch(
        background=(tf.ones([tf.shape(m_v)[0], height, width, 3]) * -1),
        vertices=vertices_view_split,
        vertex_colors=vertex_normals_split,
        faces=tf.tile(tf.cast(faces_split, tf.int32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))
    )
    temshape = tf.shape(vertex_normals_split)
    tem_vc = tf.ones(shape=temshape,dtype=np.float32)

    mask = dirt.rasterise_batch(
        background=(tf.ones([tf.shape(m_v)[0], height, width, 3]) * -1),
        vertices=vertices_view_split,
        vertex_colors=tem_vc,
        faces=tf.tile(tf.cast(faces_split, tf.int32)[np.newaxis, ...], (tf.shape(m_v)[0], 1, 1))
    )

    return normals,mask