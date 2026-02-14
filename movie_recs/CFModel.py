import numpy as np
import keras
from keras import layers, Model

class CFModel(Model):
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        super(CFModel, self).__init__(**kwargs)
        
        # --> Store the initialization arguments as instance variables
        self.n_users = n_users
        self.m_items = m_items
        self.k_factors = k_factors
        
        # User Embedding path
        self.user_embed = layers.Embedding(n_users, k_factors, name='user_embedding')
        
        # Item Embedding path
        self.item_embed = layers.Embedding(m_items, k_factors, name='item_embedding')
        
        # Flattening layers to turn (1, k) into (k,)
        self.flatten = layers.Flatten()
        
        # Dot product layer to combine the two paths
        self.dot = layers.Dot(axes=1)

    def call(self, inputs):
        # inputs is a list or tuple: [user_ids, item_ids]
        user_id, item_id = inputs
        
        user_vector = self.flatten(self.user_embed(user_id))
        item_vector = self.flatten(self.item_embed(item_id))
        
        return self.dot([user_vector, item_vector])

    def rate(self, user_id, item_id):
        # Convert inputs to tensors and predict
        prediction = self.predict([np.array([user_id]), np.array([item_id])])
        return prediction[0][0]

    # --> Added get_config method to tell Keras how to save the model parameters
    def get_config(self):
        config = super(CFModel, self).get_config()
        config.update({
            "n_users": self.n_users,
            "m_items": self.m_items,
            "k_factors": self.k_factors,
        })
        return config

    # --> Added from_config to tell Keras how to instantiate the model when loading
    @classmethod
    def from_config(cls, config):
        return cls(**config)