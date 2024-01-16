import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load pre-trained StyleGAN3 model
model_path = r"C:\Users\kunal\Desktop\face_gen\stylegan3-r-afhqv2-512x512.pkl"  # Replace with the actual path
with open(model_path, 'rb') as f:
    generator_network = pickle.load(f)

# Generate a face
def generate_face(generator_network, seed=None):
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *generator_network.input_shape[1:])
    images = generator_network.run(z, truncation_psi=0.7, randomize_noise=True, output_transform=dict(func=tf.convert_to_tensor))
    return images[0]

# Display the generated face
generated_face = generate_face(generator_network)
plt.imshow(generated_face)
plt.axis('off')
plt.show()