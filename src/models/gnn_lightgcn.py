import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse

from src.models.gcn import GCN_S
from src.models.gmf import GMF
from src.models.torch_engine import Engine

class LightGCN(torch.nn.Module):
    """Model initialisation, embedding generation and prediction of NGCF
    """

    def __init__(self, config, norm_adj):
        super(LightGCN, self).__init__()
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.emb_dim = config["emb_dim"]
        self.layer_size = config["layer_size"]
        self.n_layers = len(self.layer_size)
        self.norm_adj = norm_adj
        self.layer_size = [self.emb_dim] + self.layer_size

        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        self.init_emb()

    def dropout(self, x, keep_prob):

        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)

        return g

    def init_emb(self):
        # Initialise users and items' embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, norm_adj):
        """ Perform GNN function on users and item embeddings
        Args:
            norm_adj (torch sparse tensor): the norm adjacent matrix of the user-item interaction matrix
        Returns:
            u_g_embeddings (tensor): processed user embeddings
            i_g_embeddings (tensor): processed item embeddings
        """
        all_emb = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        embs = [all_emb]
        norm_adj = norm_adj.coalesce()
        norm_adj = norm_adj.to(self.device)

        # if self.config["dropout"]:
        #     print("droping")
        #     norm_adj = self.dropout(x=norm_adj, keep_prob=self.config["keep_pro"])
        # else:
        #     norm_adj = norm_adj
        norm_adj = self.dropout(x=norm_adj, keep_prob=self.config["keep_pro"])
        for layer in range(self.n_layers):

            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(embs, [self.n_users, self.n_items])
        return u_g_embeddings, i_g_embeddings

    def predict(self, users, items):
        """ Model prediction: dot product of users and items embeddings
        Args:
            users (int, or list of int):  user id
            items (int, or list of int):  item id
        Return:
            scores (int): dot product
        """
        users_t = torch.tensor(users, dtype=torch.int64, device=self.device)
        items_t = torch.tensor(items, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            # scores = torch.mul(
            #     self.user_embedding(users_t), self.item_embedding(items_t)
            # ).sum(dim=1)
            ua_embeddings, ia_embeddings = self.forward(self.norm_adj)
            u_g_embeddings = ua_embeddings[users_t]
            i_g_embeddings = ia_embeddings[items_t]
            scores = torch.mul(u_g_embeddings, i_g_embeddings).sum(dim=1)
        return scores


class LightGCNEngine(Engine):
    # A class includes train an epoch and train a batch of NGCF

    def __init__(self, config):
        self.config = config
        self.regs = config["regs"]  # reg is the regularisation
        self.decay = self.regs
        self.batch_size = config["batch_size"]
        self.norm_adj = config["norm_adj"]
        self.num_batch = config["num_batch"]
        self.model = LightGCN(config, self.norm_adj)
        super(LightGCNEngine, self).__init__(config)
        self.model.to(self.device)
        self.load_pretrain_weights()

    def train_single_batch(self, batch_data):
        """
        Args:
            batch_data (list): batch users, positive items and negative items
        Return:
            loss (float): batch loss
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        norm_adj = self.norm_adj.to(self.device)
        ua_embeddings, ia_embeddings = self.model.forward(norm_adj)

        batch_users, pos_items, neg_items = batch_data

        u_g_embeddings = ua_embeddings[batch_users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

        batch_loss.backward()
        self.optimizer.step()
        loss = batch_loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """ Generate batch data for each batch
        Args:
            epoch_id (int):
            train_loader (function): user, pos_items and neg_items generator
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0

        n_batch = self.num_batch

        for idx in range(n_batch):
            users, pos_items, neg_items = train_loader.sample(self.batch_size)
            batch_data = (users, pos_items, neg_items)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def load_pretrain_weights(self):

        print("loading pretrain weight")
        gcn_model = GCN_S(self.config)

        if self.config["pre_train"] == 0:
            gcn_save_dir = os.path.join(
                self.config["model_save_dir"], self.config["gcn_config"]["save_name"]
            )
        else:
            gcn_save_dir = self.config["root_dir"] + "/pre_train_weight/" + "gcn.model_" + self.config["dataset"]

        self.resume_checkpoint(
            gcn_save_dir, gcn_model,
        )
        self.model.user_embedding.weight.data = gcn_model.embedding_user.weight.data
        self.model.item_embedding.weight.data = gcn_model.embedding_item.weight.data


    def bpr_loss(self, users, pos_items, neg_items):
        # Calculate BPR loss
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = (
            1.0 / 2 * (users ** 2).sum()
            + 1.0 / 2 * (pos_items ** 2).sum()
            + 1.0 / 2 * (neg_items ** 2).sum()
        )
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss