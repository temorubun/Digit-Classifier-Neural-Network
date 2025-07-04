import network
import mnist_loader

# Create network
net = network.Network([784, 16, 16, 10])

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Train network
print("Training network...")
net.train(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Save trained model
print("Saving model...")
net.save("trained_network.pkl")
print("Done!") 