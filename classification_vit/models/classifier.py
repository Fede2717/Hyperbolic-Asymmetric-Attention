import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.poincare.manifold import CustomPoincare

from lib.lorentz.layers import LorentzMLR
from lib.poincare.layers import UnidirectionalPoincareMLR

from lib.models.ViT import (
    vit,
    lorentz_vit
)

from classification_vit.lorentz_proto_classifier import LorentzPrototypeClassifier

VIT_MODEL = {
    "euclidean" : vit,
    "lorentz" : lorentz_vit,
    #"poincare" : poincare_vit,
}

EUCLIDEAN_DECODER = {
    'mlr' : nn.Linear
}

LORENTZ_DECODER = {
    'mlr' : LorentzMLR
}

POINCARE_DECODER = {
    'mlr' : UnidirectionalPoincareMLR
}

class ViTClassifier(nn.Module):
    """ Classifier based on ViT encoder.
    """
    def __init__(self, 
            enc_type:str="lorentz", 
            dec_type:str="lorentz",
            enc_kwargs={},
            dec_kwargs={}
        ):
        super(ViTClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type

        self.clip_r = dec_kwargs['clip_r']

        self.encoder = VIT_MODEL[enc_type](remove_linear=True, **enc_kwargs)
        self.enc_manifold = self.encoder.manifold

        self.dec_manifold = None
        if dec_type == "euclidean":
            self.decoder = EUCLIDEAN_DECODER[dec_kwargs['type']](dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        elif dec_type == "lorentz":
            self.dec_manifold = CustomLorentz(k=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            if dec_kwargs.get('use_proto_softmax', False):
                self.decoder = LorentzPrototypeClassifier(
                    manifold=self.dec_manifold,
                    hidden_dim_lorentz=dec_kwargs['embed_dim'] + 1,
                    num_classes=dec_kwargs['num_classes'],
                    K=float(dec_kwargs['k']),
                    proto_seed=int(dec_kwargs.get('proto_seed', 42)),
                    d_s=float(dec_kwargs.get('d_s', 0.3)),
                    d_f_mid=float(dec_kwargs.get('d_f_mid', 1.175)),
                    T_init=float(dec_kwargs.get('T_init', 1.0)),
                    dataset_name=str(dec_kwargs.get('dataset_name', 'CIFAR-100')),
                    hierarchy_path=dec_kwargs.get('hierarchy_path'),
                )
            else:
                # PHASE2: legacy LorentzMLR decoder; preserved for ablation comparisons.
                self.decoder = LORENTZ_DECODER[dec_kwargs['type']](
                    self.dec_manifold, dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        elif dec_type == "poincare":
            self.dec_manifold = CustomPoincare(c=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            self.decoder = POINCARE_DECODER[dec_kwargs['type']](dec_kwargs['embed_dim'], dec_kwargs['num_classes'], True, self.dec_manifold)
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")
        
    def check_manifold(self, x):
        if self.enc_type=="euclidean" and self.dec_type=="euclidean":
            pass
        elif self.enc_type=="euclidean" and self.dec_type=="lorentz":
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r/x_norm)*x # Clipped HNNs
            x = self.dec_manifold.projx(F.pad(x, pad=(1,0), value=0))
        elif self.enc_type=="euclidean" and self.dec_type=="poincare":
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r/x_norm)*x # Clipped HNNs
            x = self.dec_manifold.expmap0(x)
        elif self.enc_type=="lorentz" and self.dec_type=="euclidean":
            x = self.enc_manifold.logmap0(x)[..., 1:]
        elif self.enc_manifold.k != self.dec_manifold.k:
            temp = self.dec_manifold.rescale_to_max(self.enc_manifold.logmap0(x))
            x = self.dec_manifold.expmap0(temp)
        
        return x
    
    def embed(self, x):
        x = self.encoder(x)
        embed = self.check_manifold(x)
        return embed
    
    def get_embeddings(self, x):
        return self.embed(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.check_manifold(x)
        x = self.decoder(x)
        return x
        

