import torch
import json


import copy




import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os
from tqdm import tqdm

def apply_decoder_params(z, param_vec, latent_dim, hidden_dim, output_dim):
    """
    z:         (B, latent_dim)   batch of latent features
    param_vec: (B, param_size)   per-sample decoder parameters
    returns:   decoded output of shape (B, output_dim)
    """
    B = z.size(0)
    
    # total params:
    #   W1: (hidden_dim, latent_dim)
    #   b1: (hidden_dim,)
    #   W2: (output_dim, hidden_dim)
    #   b2: (output_dim,)
    
    num_W1 = hidden_dim * latent_dim
    num_b1 = hidden_dim
    num_W2 = output_dim * hidden_dim
    num_b2 = output_dim
    
    # slice
    W1_flat = param_vec[:, 0 : num_W1]
    b1_flat = param_vec[:, num_W1 : num_W1 + num_b1]
    W2_flat = param_vec[:, num_W1 + num_b1 : num_W1 + num_b1 + num_W2]
    b2_flat = param_vec[:, num_W1 + num_b1 + num_W2 : num_W1 + num_b1 + num_W2 + num_b2]
    
    # reshape
    W1 = W1_flat.view(B, hidden_dim, latent_dim)
    b1 = b1_flat.view(B, hidden_dim)
    W2 = W2_flat.view(B, output_dim, hidden_dim)
    b2 = b2_flat.view(B, output_dim)
    
    # We do a batch matmul:
    # 1) hidden = ReLU(z W1^T + b1)
    #    z shape:    (B, 1, latent_dim)
    #    W1^T shape: (B, latent_dim, hidden_dim)
    
    # Expand z to (B, 1, latent_dim) for bmm
    z_expanded = z.unsqueeze(1)            # (B, 1, latent_dim)
    W1_t = W1.transpose(1, 2)              # (B, latent_dim, hidden_dim)
    hidden = torch.bmm(z_expanded, W1_t)   # => (B, 1, hidden_dim)
    hidden = hidden.squeeze(1) + b1        # => (B, hidden_dim)
    hidden = torch.relu(hidden)
    
    # 2) out = hidden W2^T + b2 => shape (B, output_dim)
    hidden_expanded = hidden.unsqueeze(1)         # (B, 1, hidden_dim)
    W2_t = W2.transpose(1, 2)                     # (B, hidden_dim, output_dim)
    out = torch.bmm(hidden_expanded, W2_t).squeeze(1)  # (B, output_dim)
    out = out + b2
    
    return out


def apply_encoder_params(x, param_vec, input_dim, hidden_dim, latent_dim):
    """
    x:         (B, input_dim)     batch of input features
    param_vec: (B, param_size)    per-sample encoder parameters
    returns:   mu, logvar shapes (B, latent_dim) each
    """
#    z = z.to(device)
 #   param_vec = param_vec.to(device)
    B = x.size(0)
    
    # total params:
    #   W1: (hidden_dim, input_dim)
    #   b1: (hidden_dim,)
    #   W2: (2*latent_dim, hidden_dim)  # 2*latent_dim => (mu + logvar) in final layer
    #   b2: (2*latent_dim,)
    
    num_W1 = hidden_dim * input_dim
    num_b1 = hidden_dim
    num_W2 = (2*latent_dim) * hidden_dim
    num_b2 = 2*latent_dim
    
    # slice
    W1_flat = param_vec[:, 0 : num_W1]
    b1_flat = param_vec[:, num_W1 : num_W1+num_b1]
    W2_flat = param_vec[:, num_W1+num_b1 : num_W1+num_b1+num_W2]
    b2_flat = param_vec[:, num_W1+num_b1+num_W2 : num_W1+num_b1+num_W2+num_b2]
    
    # reshape
    W1 = W1_flat.view(B, hidden_dim, input_dim)
    b1 = b1_flat.view(B, hidden_dim)
    W2 = W2_flat.view(B, 2*latent_dim, hidden_dim)
    b2 = b2_flat.view(B, 2*latent_dim)
    
    # We do a batch matmul:
    # 1) hidden = ReLU(x W1^T + b1)
    #    x shape:    (B, 1, input_dim)
    #    W1^T shape: (B, input_dim, hidden_dim)
    
    # Expand x to (B, 1, input_dim) for bmm
    x_expanded = x.unsqueeze(1)            # (B, 1, input_dim)
    W1_t = W1.transpose(1,2)              # (B, input_dim, hidden_dim)
    hidden = torch.bmm(x_expanded, W1_t)  # => (B, 1, hidden_dim)
    hidden = hidden.squeeze(1) + b1       # => (B, hidden_dim)
    hidden = torch.relu(hidden)
    
    # 2) out = hidden W2^T + b2 => shape (B, 2*latent_dim)
    hidden_expanded = hidden.unsqueeze(1)         # (B, 1, hidden_dim)
    W2_t = W2.transpose(1,2)                      # (B, hidden_dim, 2*latent_dim)
    out = torch.bmm(hidden_expanded, W2_t).squeeze(1)  # (B, 2*latent_dim)
    out = out + b2
    
    # split into mu, logvar
    mu = out[:, :latent_dim]
    logvar = out[:, latent_dim:]
    return mu, logvar

# ------------------------------------------------------------------
# 2) HYPERNET: Maps mask -> encoder parameter vector
# ------------------------------------------------------------------

class HynetImpute():
    def __init__(self, latent_dim):
        self.latent_dim=latent_dim
        self.batch_size=16
        hidden_dim_enc = latent_dim
        self.hyper_hidden_dim = max(1, latent_dim // 2)
        self.hidden_dim_enc =  latent_dim  # sub-network hidden
        self.hidden_dim_dec = latent_dim
    
    def train(self,train_data,train_mask,train_ground_truth):
        input_dim=train_data.size(1)
        self.model=train_complete_hypernet(train_data,train_mask,train_ground_truth,input_dim,self.hyper_hidden_dim,self.batch_size,self.hidden_dim_enc,self.hidden_dim_dec,self.latent_dim,gamma=0.2)


        
    def predict(self,test_data,test_mask):
        input_dim=test_data.size(1)
        test_recon=hynetimpute(self.model,test_data, test_mask, input_dim,self.hidden_dim_enc, self.latent_dim)
        return test_recon



class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, dec_hidden_dim, out_dim):
        super().__init__()
        self.decoder  = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden_dim),
            nn.ReLU(),
            nn.Linear(dec_hidden_dim, out_dim)  # output dimension = input_dim
        ).to(device)
    
    def forward(self, z):
        return self.decoder(z)

class HyperNet(nn.Module):
    def __init__(self, d_mask, hyper_hidden_dim, enc_param_size):
        """
        d_mask: dimension of mask (== input_dim for mask)
        hyper_hidden_dim: hidden layer size in hypernet
        enc_param_size: total # of parameters needed for the encoder sub-network
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mask, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, enc_param_size)
        ).to(device)
    
    def forward(self, mask):
        """
        mask: (B, d_mask)
        returns param_vec shape (B, enc_param_size)
        """
        return self.net(mask.float().to(device))  # ensure float in case mask is 0/1





def train_hypernet(model, dataloader,input_dim,hidden_dim_enc,latent_dim,  epochs=50, lr=1e-3, missing_rate=0.2,is_train_by_sythenticmissing=False,gamma=0.5):
    hypernet=model['hypernet']
    decoder=model['decoder']

    #layer1 = hypernet.net[0]  # first Linear
    #layer2 = hypernet.net[2]  # second Linear
     # optimizer
    optimizer = optim.Adam(list(hypernet.parameters()) + list(decoder.parameters()), lr=lr)
    
    reconstruction_loss_fn = nn.MSELoss(reduction='none')  # Reconstruction loss without reduction
    # training loop
    
    decide_syn_miss=1.0
    if is_train_by_sythenticmissing==True:decide_syn_miss=gamma
    for ep in range(epochs):
        total_loss = 0.0
        kl_total_loss=0.0
        for x_batch, m_batch,_ in dataloader:


            is_simulated = random.uniform(0, 1)  
            if is_simulated>decide_syn_miss:
                simulated_data,simulated_mask = apply_simulated_missingness(x_batch, m_batch, missing_rate)
            else:
                simulated_data,simulated_mask = x_batch, m_batch
            # x_batch: (B, input_dim)
            # m_batch: (B, input_dim)
            optimizer.zero_grad()
            
            # 1) hypernet -> param_vec for each sample
            param_vecs = hypernet(simulated_mask)   # (B, enc_param_size)
            
     
            mu, logvar = apply_encoder_params(simulated_data, param_vecs, input_dim,hidden_dim_enc, latent_dim)
            # reparameterize
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # 3) decode
            x_recon = decoder(z)
            
            # 4) compute VAE loss

            recon_loss = (reconstruction_loss_fn(x_recon, x_batch) * m_batch).sum() / m_batch.sum()
            
            # KL( q(z|x) || p(z) ) = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
            kl = 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
            loss=recon_loss+kl
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() 
            kl_total_loss+=kl.item()
        
        avg_loss = total_loss 
        if ep%10==0:
            print(f"Epoch {ep+1}/{epochs}, Loss: {total_loss:.4f},kl loss:{kl_total_loss:.4f}")

def hynetimpute(model,data, mask, input_dim,hidden_dim_enc, latent_dim):

    hypernet=model['hypernet']
    decoder=model['decoder']
    param_vecs = hypernet(mask)   # (B, enc_param_size)
    
    # 2) apply encoder sub-net
    mu, logvar = apply_encoder_params(data, param_vecs, input_dim,hidden_dim_enc, latent_dim)
    # reparameterize
    z = mu 
    
    # 3) decode
    x_recon = decoder(z)
    return x_recon

def extract_encoder_params(vae):
    """
    Flatten the encoder's 2 linear layers into a single parameter vector.
    Returns a torch 1D tensor param_vec of shape (enc_param_size,).
    """
    # encoder[0] => Linear(...), encoder[2] => Linear(...)
    # Note: We skip the ReLU layer
    layer1 = vae.encoder[0]
    layer2 = vae.encoder[2]

    # Flatten
    W1 = layer1.weight.data.view(-1)
    b1 = layer1.bias.data.view(-1)
    W2 = layer2.weight.data.view(-1)
    b2 = layer2.bias.data.view(-1)

    return torch.cat([W1, b1, W2, b2], dim=0)
def initialize_hypernet(hypernet):
    """
    Force hypernet to produce cvae_param_vec for *any* mask input
    by zeroing out weights and copying cvae_param_vec into the final bias.
    """
    # The net has 3 layers in nn.Sequential: [Linear, ReLU, Linear].
    layer1 = hypernet.net[0]  # first Linear
    layer2 = hypernet.net[2]  # second Linear

    with torch.no_grad():
        # zero the weights in both linear layers
        layer1.weight.zero_()
        layer1.weight.normal_(mean=0, std=1e-2)
        layer1.bias.zero_()
        layer1.bias.normal_(mean=0, std=1e-2)
        layer2.weight.zero_()
        layer2.weight.normal_(mean=0, std=1e-2)
        layer2.bias.zero_()
        layer2.bias.normal_(mean=0, std=1e-2)






def apply_simulated_missingness(data, mask, missing_rate=0.2):
    simulated_mask = mask.clone()
    simulated_data = data.clone()
    additional_mask = torch.rand_like(simulated_mask) > missing_rate
    simulated_mask = simulated_mask * additional_mask
    simulated_data=simulated_data*additional_mask
    return simulated_data,simulated_mask







def train_complete_hypernet(train_data,train_mask,train_ground_truth,input_dim,hyper_hidden_dim,batch_size,hidden_dim_enc,hidden_dim_dec,latent_dim,gamma=0.5):
    train_dataset = torch.utils.data.TensorDataset(train_data, train_mask,train_ground_truth)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    enc_param_size = (hidden_dim_enc * input_dim) + hidden_dim_enc \
                     + ((2*latent_dim) * hidden_dim_enc) + (2*latent_dim)
    
    hypernet = HyperNet(d_mask=input_dim, hyper_hidden_dim=hyper_hidden_dim,
                        enc_param_size=enc_param_size).to(device)
    decoder = SharedDecoder(latent_dim, hidden_dim_dec, input_dim).to(device)

    initialize_hypernet(hypernet)
    #    a=1
    model = {"hypernet": hypernet, "decoder": decoder}
    train_hypernet(model, train_dataloader,input_dim,hidden_dim_enc,latent_dim,  epochs=100, lr=0.001,gamma=gamma)
    return model




 

# Main Code
if __name__ == "__main__":

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    hynetImputer=HynetImpute(latent_dim=10)
    #make a simple data and demonstarte the following hynetImputer
    groun_truth=torch.randn(100,10)
    mask=torch.randint(0,2,(100,10))
    data=groun_truth*mask
    ##splict train and test
    train_data=data[:80]
    train_mask=mask[:80]
    train_ground_truth=groun_truth[:80]
    test_data=data[80:]
    test_mask=mask[80:]
    test_ground_truth=groun_truth[80:]

    hynetImputer.train(train_data,train_mask,train_ground_truth)
    imputed_data=hynetImputer.predict(test_data,test_mask)
    print(imputed_data)
    








