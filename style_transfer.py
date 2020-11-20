import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import helpers.images.offline_image_helper as offline_image_helper
from helpers.plotting import basic_plotting
from tensorflow.keras.applications import vgg19
from IPython.display import display, clear_output
from tkinter import Tk
from tkinter.filedialog import askopenfilename

window = Tk()
style_loc = askopenfilename(title='Choose your style', filetypes=[('jpeg files', '.jpg')])
image_loc = askopenfilename(title='Choose your image', filetypes=[('jpeg files', '.jpg')])
window.destroy()

style = offline_image_helper.load_image(style_loc, target_size=(500, 500))
img = offline_image_helper.load_image(image_loc, target_size=(500, 500))
basic_plotting.plot_image_grid([img, style])
plt.show()

base_model = vgg19.VGG19(include_top=False, weights="imagenet", )

base_model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
#                              [(no. images, hight, widht, chanels: 3)]
# input_1 (InputLayer)          [(None, None, None, 3)]   0
# RGB because have 3 channels
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, None, None, 64)    1792
# convolutional neural network, takes 64 filters, will made 1792 parameters

style_img = style
content_img = img

# VGG19 will take the image and rescale it, and flip R and B, so instead of RGB we have BGR

processed_style = vgg19.preprocess_input(np.expand_dims(style_img, axis=0))
# The colors have been flipped, and have been subtracted
processed_content = vgg19.preprocess_input(np.expand_dims(content_img, axis=0))

print(processed_style.shape)
print(processed_content.shape)

basic_plotting.plot_image_grid([processed_content[0], processed_style[0]])
plt.show()

VGG_BIASES = vgg19.preprocess_input((np.zeros(3)).astype("float32"))
print(VGG_BIASES)


# we dont want the output that has been processed, we want the real image, so we do un-processing, and flip
# the blue and red channels to be RGB again

def deprocess(processed_img):
    unprocessed_img = processed_img - VGG_BIASES
    # Unstack it to get 1,152,152,3
    unprocessed_img = tf.unstack(unprocessed_img, axis=-1)
    # Stack it again to be 3,1,152,152
    unprocessed_img = tf.stack([unprocessed_img[2], unprocessed_img[1], unprocessed_img[0]], axis=-1)
    return unprocessed_img


# So now instead of going to 0-1 it goes 0-255
plt.imshow(np.round(deprocess(processed_content)[0]) / 255)
plt.show()

# In style transfer we just one to take certain elements of the network, and those to be the inputs to our content
# loss and style loss

# this is going to be the layer we are going to be using to our content loss
CONTENT_LAYERS = ["block5_conv4"]
# the blok 3 and 2 are too early so its not very illustrative
# and the 5th (the last one) cause it takes out all the content
STYLE_LAYERS = ["block4_conv1", "block4_conv2", "block4_conv3", "block4_conv4"]


def make_model():
    base_model = vgg19.VGG19(include_top=False, weights='imagenet')
    base_model.trainable = False
    content_layers = CONTENT_LAYERS
    style_layers = STYLE_LAYERS
    output_layers = [base_model.get_layer(layer).output for layer in (content_layers + style_layers)]
    return tf.keras.models.Model(base_model.input, output_layers)


base_model = make_model()
content_outputs = base_model(processed_content)
style_outputs = base_model(processed_style)

image_content = content_outputs[0]
style_content = style_outputs[0]


def get_content_loss(new_image_content, base_image_content):
    return np.mean(np.square(new_image_content - base_image_content))


def get_gram_matrix(output):
    # our style_layer has 4 layers, so we take one of those, the first one
    first_style_layer = output
    style_layer = tf.reshape(first_style_layer, (-1, first_style_layer.shape[-1]))
    n = style_layer.shape[0]
    # matrix multiplication of each of the filters after processing
    gram_matrix = tf.matmul(style_layer, style_layer, transpose_a=True)
    n = gram_matrix.shape[0]
    # we get the gram matrix and normalize it, and the size of the square matrix
    return gram_matrix / tf.cast(n, "float32"), n


gram_matrix, N = get_gram_matrix(style_outputs[2])
plt.figure(figsize=(10, 10))
plt.imshow(gram_matrix.numpy())
plt.show()


# So this in essence is the style of the image, one of the elements of the style
# the style is represented by the relationship between the filters and not themselves, its how they interact
# each other rather just the content


def get_style_loss(new_image_style, base_style):
    new_style_gram, new_gram_num_height = get_gram_matrix(new_image_style)
    base_style_gram, base_gram_num_height = get_gram_matrix(base_style)
    # make sure both are same size to be possible to subtract them
    assert new_gram_num_height == base_gram_num_height
    gram_num_features = new_gram_num_height
    loss = tf.reduce_sum(
        tf.square(base_style_gram - new_style_gram) / (4 * (new_gram_num_height ** 2) * (gram_num_features ** 2)))
    return loss


def get_total_loss(new_image_output, base_content_image_output, base_style_image_output, alpha=.999):
    new_image_styles = new_image_output[len(CONTENT_LAYERS):]
    base_image_styles = base_style_image_output[len(CONTENT_LAYERS):]
    style_loss = 0
    N = len(new_image_styles)
    for i in range(N):
        style_loss += get_style_loss(new_image_styles[i], base_image_styles[i])

    new_image_contents = new_image_output[:len(CONTENT_LAYERS)]
    base_image_contents = base_content_image_output[:len(CONTENT_LAYERS)]
    content_loss = 0
    N = len(new_image_contents)
    for i in range(N):
        content_loss += get_content_loss(new_image_contents[i], base_image_contents[i]) / N

    return (1 - alpha) * style_loss + alpha * content_loss


get_total_loss(style_outputs, content_outputs, style_outputs)


# ---------------------
#       TRAINING
# ---------------------

base_style_outputs = base_model(processed_style)
base_content_output = base_model(processed_content)
processed_content_var = tf.Variable(processed_content + tf.random.normal(processed_content.shape))
optimizer = tf.optimizers.Adam(5, beta_1=.99, epsilon=1e-3)

images = []
losses = []

# number of steps we take
i = 0
best_loss = 9999999
min_vals = VGG_BIASES
max_vals = 255 + VGG_BIASES

n = int(input("How many iteration do you want? : "))

for i in range(n):
    # help us to differentiate our losses with respect to our variables
    with tf.GradientTape() as tape:
        tape.watch(processed_content_var)
        content_var_outputs = base_model(processed_content_var)
        loss = get_total_loss(content_var_outputs, base_content_output, base_style_outputs, alpha=.97)
        grad = tape.gradient(loss, processed_content_var)
        losses.append(loss)
        optimizer.apply_gradients(zip([grad], [processed_content_var]))
        clipped = tf.clip_by_value(processed_content_var, min_vals, max_vals)
        processed_content_var.assign(clipped)
        if i % 20:
            images.append(deprocess(processed_content_var))
        if loss < best_loss:
            best_image = processed_content_var
            best_loss = loss
        display(loss)
        clear_output(wait=True)
    print(i)

unprocessed_best_image = deprocess(best_image)
plt.figure(figsize=(10, 10))
plt.imshow(unprocessed_best_image[0] / 255)
plt.show()
