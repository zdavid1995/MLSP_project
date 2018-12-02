import numpy as np
import matplotlib.pyplot as plt

dec_acc_cosine = np.load("decoder_accuracies_cosine.npy")
dec_acc_l2 = np.load("decoder_accuracies_L2.npy")

dec_losses_cosine = np.load("decoder_losses_cosine.npy")
dec_losses_l2 = np.load("decoder_losses_L2.npy")

enc_losses_cosine = np.load("encoder_losses_cosine.npy")
enc_losses_l2 = np.load("encoder_losses_L2.npy")

fig = plt.figure()
plt.plot(dec_acc_cosine,label="cosine")
plt.plot(dec_acc_l2,label="L2")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Decoder Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
plt.close(fig)

fig = plt.figure()
plt.plot(dec_losses_cosine,label="cosine")
plt.plot(dec_losses_l2,label="L2")
plt.xlabel("Step")
plt.ylabel("Cross Entropy Loss")
plt.title("Decoder Cross Entropy Loss")
plt.legend()
plt.savefig("decoder_loss_plot.png")
plt.close(fig)

fig = plt.figure()
plt.plot(enc_losses_cosine)
plt.xlabel("Step")
plt.ylabel("Cosine Distance")
plt.title("Encoder Cosine Loss")
plt.savefig("encoder_loss_cosine_plot.png")
plt.close(fig)

fig = plt.figure()
plt.plot(enc_losses_l2)
plt.xlabel("Step")
plt.ylabel("MSE Distance")
plt.title("Encoder L2 Loss")
plt.savefig("encoder_loss_l2_plot.png")
plt.close(fig)
