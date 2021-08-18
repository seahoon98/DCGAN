import torch
import torch.nn as nn

'''
Jointly train generator G and discriminator D with a minimax game
'''

D = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid())

# Generator network 
G = nn.Sequential(
        nn.Linear(100, 128),
        nn.ReLU(),
        nn.Linear(128, 28*28),
        nn.Tanh())

loss = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), 1e-2)
g_optimizer = torch.optim.Adam(G.parameters(), 1e-2)

while True:
    # let x be real samples of shape (batch, 28*28)
    x = torch.rand(28 * 28, 128)
    # let z be latent variables of shape (batch, 100)
    z = torch.rand(100, 128)
    # torch.transpose(z, 0, 1)
    # torch.transpose(x, 0, 1)

    # Gradient ascent on discriminator
    # update D
    d_loss = loss(D(x), 1) + loss(D(G(z)), 0)
    d_loss.backward()
    d_optimizer.step()

    # Instead: Gradient ascent on generator, different objective
    # update G
    g_loss = loss(D(G(z)), 1)
    g_loss.backword()
    g_optimizer.step()
    
