import numpy as np
import phe as paillier
import json
from time import time
import matplotlib.pyplot as plt

ENCRYPT = False

# ref: https://arxiv.org/pdf/1906.05108.pdf

class User:
    def __init__(self, user_id, initial_profile, actual_ratings, mask, lr, pubkey, privkey, predicted_ratings):
        self.user_id = user_id
        self.user_profile = initial_profile
        self.actual_ratings = actual_ratings  # actual ratings for each movie
        self.predicted_ratings = predicted_ratings  # predicted ratings for each movie
        self.mask = mask  # binary mask indicating whether the user has rated each movie
        self.lr = lr
        self.pubkey = pubkey
        self.privkey = privkey

    def compute_gradient(self, item_profile, encrypt=False):
        if encrypt:
            # array zeros with same shape as item profile
            new_item_profile = np.zeros(item_profile.shape, dtype=np.float64)
            # decrypt item profile
            for i in range(len(item_profile)):
                for j in range(len(item_profile[i])):
                    new_item_profile[i][j] = self.privkey.decrypt(item_profile[i][j])

            item_profile = new_item_profile

        # Compute predicted rating for each movie
        self.predicted_ratings = np.dot(self.user_profile, item_profile.T)

        # Compute gradient based on difference between actual and predicted ratings
        gradient = 2 * np.dot((self.predicted_ratings - self.actual_ratings) * self.mask, item_profile)

        x = self.lr * gradient
        
        # update user profile based on gradient
        self.user_profile = self.user_profile - (self.lr * gradient)

        if encrypt:
            # encrypt gradient
            gradient = np.array([self.pubkey.encrypt(x) for x in gradient])

        return gradient, self.compute_loss(self.predicted_ratings)

    def compute_loss(self, predicted_ratings):
        return np.sum(((self.actual_ratings - predicted_ratings) * self.mask) ** 2)
    
    def updateRating(self, movieId, rating):
        self.actual_ratings[movieId] = rating
        if rating == 0:
          self.mask[movieId] = 0
        else:
          self.mask[movieId] = 1

    def getInfo(self):
        print(f"User {self.user_id} info:")
        print(f"actual ratings: {self.actual_ratings * self.mask}")
        print(f"predicted ratings: {np.rint(self.predicted_ratings * self.mask)}")


class Server:
    def __init__(self, initial_item_profile, lr, num_users, pubkey):
        self.item_profile = initial_item_profile
        self.lr = lr
        self.num_users = num_users
        self.pubkey = pubkey

    def update_item_profile(self, user_gradient):
        # Update item profile based on user gradient
        self.item_profile -= (self.lr / self.num_users) * user_gradient


def updateMatrices(users, server):
    # Set convergence threshold
    convergence_iterations_threshold = 5

    iteration = 0

    prev_loss = 0

    local_convergence = 0

    # array for storing loss for each iteration
    lossArray = []

    # Iterate over multiple rounds of gradient computation and updates
    while True:
        startTime = time()
        total_loss = 0
        for user in users:
            # Users compute gradients for each movie
            gradient, loss = user.compute_gradient(server.item_profile, ENCRYPT)
            total_loss += loss
            # Server updates item profile for the corresponding movie
            server.update_item_profile(gradient)

        # Check for convergence on loss to 3 decimal places
        if np.abs(total_loss - prev_loss) < 0.001:
            local_convergence += 1
            if local_convergence > convergence_iterations_threshold:
                print('Converged after {} iterations'.format(iteration))
                break
        else:
            local_convergence = 0
            prev_loss = total_loss

        print(f"Iteration: {iteration} | loss: {total_loss} | time: {time() - startTime}")
        lossArray.append(total_loss)

        iteration += 1

    # draw loss graph
    plt.plot(lossArray)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()


def loadData(pubkey, privkey, encrypt=False):
    data = {}
    with open('data.json') as f:
        data = json.load(f)
    
    users = []
    for userData in data["users"]:
        user = User(userData["user_id"], np.array(userData["initial_profile"]), np.array(userData["actual_ratings"]), np.array(userData["mask"]), data["server"]["lr"], pubkey, privkey, np.array(userData["predicted_ratings"]))
        users.append(user)

    item_profile = data["server"]["item_profile"]

    if encrypt:
        # encrypt item profile and save in np array
        for i in range(len(item_profile)):
            for j in range(len(item_profile[i])):
                item_profile[i][j] = pubkey.encrypt(item_profile[i][j])

    item_profile = np.array(item_profile)

    server = Server(item_profile, data["server"]["lr"], len(users), pubkey)

    return users, server


def getDirections():
    return """
    Enter a command:
    [1] Get users predicted ratings and actual ratings
    [2] Update user rating for a movie
    [3] Exit
    """


if __name__ == "__main__":
    # Initialize public and private keys for Paillier cryptosystem
    # public key is used for encryption, private key is used for decryption
    print("Generating public and private keys...")
    pubkey, privkey = paillier.generate_paillier_keypair()

    print("Loading data...")
    users, server = loadData(pubkey, privkey, ENCRYPT)

    # get input from user in a while loop
    while True:
        print(getDirections())
        command = int(input())
        if command == 1:
            for user in users:
                print("--------------------------------------------------")
                user.getInfo()
                print("--------------------------------------------------")
        elif command == 2:
            userId = int(input("Enter user id: "))
            movieId = int(input("Enter movie id: "))
            rating = int(input("Enter rating [0, 10]: "))
            users[userId].updateRating(movieId, rating)
            print("Updating user rating...")
            updateMatrices(users, server)
            print("Done!")
        elif command == 3:
            break
