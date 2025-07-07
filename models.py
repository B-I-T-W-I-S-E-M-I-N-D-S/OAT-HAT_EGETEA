# import numpy as np
# import torch
# import math
# from torch.autograd import Variable
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.nn import init


# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float = 0.1,
#                  maxlen: int = 750):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)

#     def forward(self, token_embedding: torch.Tensor):
#         return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# class AdaptiveSpanAttention(nn.Module):
#     """
#     Lightweight Adaptive Span Attention with efficient multi-range dependency modeling.
#     """
#     def __init__(self, embedding_dim, num_heads, dropout, max_span=200, min_span=8):
#         super(AdaptiveSpanAttention, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.max_span = max_span
#         self.min_span = min_span
        
#         # Simplified dual-range attention (local + global)
#         self.local_attention = nn.MultiheadAttention(
#             embed_dim=embedding_dim, num_heads=num_heads // 2, dropout=dropout
#         )
#         self.global_attention = nn.MultiheadAttention(
#             embed_dim=embedding_dim, num_heads=num_heads // 2, dropout=dropout
#         )
        
#         # Lightweight span predictor
#         self.span_predictor = nn.Sequential(
#             nn.Linear(embedding_dim, embedding_dim // 2),
#             nn.GELU(),
#             nn.Linear(embedding_dim // 2, 2),  # Just local and global
#             nn.Softmax(dim=-1)
#         )
        
#         # Single efficient relevance scorer
#         self.relevance_scorer = nn.Sequential(
#             nn.Linear(embedding_dim * 2, embedding_dim // 2),
#             nn.GELU(),
#             nn.Linear(embedding_dim // 2, 1),
#             nn.Sigmoid()
#         )
        
#         # Simplified temporal decay
#         self.temporal_decay_local = nn.Parameter(torch.tensor(0.9))
#         self.temporal_decay_global = nn.Parameter(torch.tensor(0.8))
        
#         # Simple fusion
#         self.fusion = nn.Sequential(
#             nn.Linear(embedding_dim * 2, embedding_dim),
#             nn.GELU()
#         )
        
#         # Lightweight gating
#         self.gate = nn.Linear(embedding_dim, embedding_dim)
        
#         self.layer_norm = nn.LayerNorm(embedding_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, long_x, encoded_x):
#         batch_size = long_x.shape[1]
#         seq_len = long_x.shape[0]
        
#         # Simple context representation
#         current_context = torch.mean(encoded_x, dim=0, keepdim=True)
        
#         # Predict local vs global attention weights
#         span_weights = self.span_predictor(current_context).squeeze(0)  # [batch, 2]
        
#         # Process only two ranges for efficiency
#         local_span = min(self.min_span * 4, seq_len)
#         global_span = min(self.max_span, seq_len)
        
#         local_features = []
#         global_features = []
        
#         for b in range(batch_size):
#             # Local processing
#             local_start = max(0, seq_len - local_span)
#             local_history = long_x[local_start:, b:b+1, :]
            
#             if local_history.shape[0] > 0:
#                 local_context = current_context[:, b:b+1, :].expand(local_history.shape[0], -1, -1)
#                 local_combined = torch.cat([local_history, local_context], dim=-1)
#                 local_scores = self.relevance_scorer(local_combined).squeeze(-1).squeeze(-1)
                
#                 # Apply temporal decay
#                 local_positions = torch.arange(local_history.shape[0], device=long_x.device, dtype=torch.float)
#                 local_decay = self.temporal_decay_local ** (local_history.shape[0] - 1 - local_positions)
#                 local_weights = F.softmax(local_scores + torch.log(local_decay + 1e-8), dim=0)
                
#                 local_feat = (local_history.squeeze(1) * local_weights.unsqueeze(-1)).sum(dim=0)
#             else:
#                 local_feat = torch.zeros(self.embedding_dim, device=long_x.device)
            
#             # Global processing (subsampled for efficiency)
#             if seq_len > 32:
#                 # Subsample for global context
#                 step = max(1, seq_len // 32)
#                 global_history = long_x[::step, b:b+1, :]
#             else:
#                 global_history = long_x[:, b:b+1, :]
            
#             if global_history.shape[0] > 0:
#                 global_context = current_context[:, b:b+1, :].expand(global_history.shape[0], -1, -1)
#                 global_combined = torch.cat([global_history, global_context], dim=-1)
#                 global_scores = self.relevance_scorer(global_combined).squeeze(-1).squeeze(-1)
                
#                 global_positions = torch.arange(global_history.shape[0], device=long_x.device, dtype=torch.float)
#                 global_decay = self.temporal_decay_global ** (global_history.shape[0] - 1 - global_positions)
#                 global_weights = F.softmax(global_scores + torch.log(global_decay + 1e-8), dim=0)
                
#                 global_feat = (global_history.squeeze(1) * global_weights.unsqueeze(-1)).sum(dim=0)
#             else:
#                 global_feat = torch.zeros(self.embedding_dim, device=long_x.device)
            
#             local_features.append(local_feat)
#             global_features.append(global_feat)
        
#         # Combine features
#         local_stack = torch.stack(local_features, dim=0)
#         global_stack = torch.stack(global_features, dim=0)
        
#         # Weighted combination
#         combined = torch.cat([
#             local_stack * span_weights[:, 0:1],
#             global_stack * span_weights[:, 1:2]
#         ], dim=-1)
        
#         # Simple fusion
#         fused = self.fusion(combined)
        
#         # Apply gating
#         gate_weights = torch.sigmoid(self.gate(fused))
#         adaptive_features = fused * gate_weights
        
#         # Generate attention weights (simplified)
#         attention_weights = torch.ones(seq_len, batch_size, device=long_x.device) / seq_len
#         actual_spans = [local_span, global_span]
        
#         return attention_weights, adaptive_features, actual_spans


# class HierarchicalContextEncoder(nn.Module):
#     """
#     Lightweight hierarchical encoder with efficient multi-scale processing.
#     """
#     def __init__(self, embedding_dim, num_heads, dropout):
#         super(HierarchicalContextEncoder, self).__init__()
        
#         # Reduced encoder layers
#         self.fine_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=embedding_dim, nhead=num_heads, 
#                 dropout=dropout, activation='gelu',
#                 batch_first=False
#             ), num_layers=2  # Reduced from 3
#         )
        
#         self.coarse_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=embedding_dim, nhead=num_heads // 2,  # Reduced heads
#                 dropout=dropout, activation='gelu',
#                 batch_first=False
#             ), num_layers=1  # Reduced from 2
#         )
        
#         # Simplified fusion
#         self.scale_fusion = nn.Sequential(
#             nn.Linear(embedding_dim * 2, embedding_dim),
#             nn.GELU()
#         )
        
#         # Lightweight temporal consistency
#         self.temporal_consistency = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        
#         # Simple feature selection
#         self.feature_selector = nn.Sequential(
#             nn.Linear(embedding_dim * 2, embedding_dim // 2),
#             nn.GELU(),
#             nn.Linear(embedding_dim // 2, 2),
#             nn.Softmax(dim=-1)
#         )
        
#         self.layer_norm = nn.LayerNorm(embedding_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, features, attention_mask):
#         seq_len, batch_size, dim = features.shape
        
#         # Fine-grained processing
#         fine_features = self.fine_encoder(features)
        
#         # Coarse-grained processing (only if sequence is long enough)
#         if seq_len >= 4:
#             # Downsample
#             coarse_features = features.permute(1, 2, 0)  # [batch, dim, seq]
#             coarse_features = F.avg_pool1d(coarse_features, kernel_size=4, stride=4, padding=0)
#             coarse_features = coarse_features.permute(2, 0, 1)  # [seq, batch, dim]
            
#             # Encode
#             coarse_encoded = self.coarse_encoder(coarse_features)
            
#             # Upsample
#             coarse_upsampled = coarse_encoded.permute(1, 2, 0)  # [batch, dim, seq]
#             coarse_upsampled = F.interpolate(
#                 coarse_upsampled, size=seq_len, mode='linear', align_corners=False
#             )
#             coarse_upsampled = coarse_upsampled.permute(2, 0, 1)  # [seq, batch, dim]
#         else:
#             coarse_upsampled = fine_features
        
#         # Simple feature selection
#         combined_for_selection = torch.cat([fine_features, coarse_upsampled], dim=-1)
#         selection_weights = self.feature_selector(combined_for_selection)
        
#         # Weighted combination
#         selected_features = (fine_features * selection_weights[:, :, 0:1] + 
#                            coarse_upsampled * selection_weights[:, :, 1:2])
        
#         # Lightweight temporal consistency
#         consistency_features = selected_features.permute(1, 2, 0)  # [batch, dim, seq]
#         consistency_features = self.temporal_consistency(consistency_features)
#         consistency_features = consistency_features.permute(2, 0, 1)  # [seq, batch, dim]
        
#         # Final combination
#         output = selected_features + consistency_features
#         output = self.layer_norm(output)
        
#         return output


# class MYNET(torch.nn.Module):
#     def __init__(self, opt):
#         super(MYNET, self).__init__()
#         self.n_feature=opt["feat_dim"] 
#         n_class=opt["num_of_class"]
#         n_embedding_dim=opt["hidden_dim"]
#         n_enc_layer=opt["enc_layer"]
#         n_enc_head=opt["enc_head"]
#         n_dec_layer=opt["dec_layer"]
#         n_dec_head=opt["dec_head"]
#         n_seglen=opt["segment_size"]
#         self.anchors=opt["anchors"]
#         self.anchors_stride=[]
#         dropout=0.3
#         self.best_loss=1000000
#         self.best_map=0
        
#         # Enhanced feature reduction with normalization
#         self.feature_reduction_rgb = nn.Sequential(
#             nn.Linear(self.n_feature//2, n_embedding_dim//2),
#             nn.GELU(),
#             nn.LayerNorm(n_embedding_dim//2)
#         )
#         self.feature_reduction_flow = nn.Sequential(
#             nn.Linear(self.n_feature//2, n_embedding_dim//2),
#             nn.GELU(),
#             nn.LayerNorm(n_embedding_dim//2)
#         )
        
#         self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
#         # Add adaptive span attention for long-range dependencies
#         self.adaptive_span_attention = AdaptiveSpanAttention(
#             embedding_dim=n_embedding_dim,
#             num_heads=4,
#             dropout=dropout,
#             max_span=200,
#             min_span=8
#         )
        
#         # Add hierarchical context encoder
#         self.hierarchical_encoder = HierarchicalContextEncoder(
#             embedding_dim=n_embedding_dim,
#             num_heads=4,
#             dropout=dropout
#         )
        
#         self.encoder = nn.TransformerEncoder(
#                                             nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
#                                                                         nhead=n_enc_head, 
#                                                                         dropout=dropout, 
#                                                                         activation='gelu'), 
#                                             n_enc_layer, 
#                                             nn.LayerNorm(n_embedding_dim))
                                            
#         self.decoder = nn.TransformerDecoder(
#                                             nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
#                                                                         nhead=n_dec_head, 
#                                                                         dropout=dropout, 
#                                                                         activation='gelu'), 
#                                             n_dec_layer, 
#                                             nn.LayerNorm(n_embedding_dim))
                                            
#         # Enhanced classification and regression heads
#         self.classifier = nn.Sequential(
#             nn.Linear(n_embedding_dim, n_embedding_dim), 
#             nn.GELU(), 
#             nn.LayerNorm(n_embedding_dim),
#             nn.Dropout(dropout),
#             nn.Linear(n_embedding_dim, n_class)
#         )
#         self.regressor = nn.Sequential(
#             nn.Linear(n_embedding_dim, n_embedding_dim), 
#             nn.GELU(), 
#             nn.LayerNorm(n_embedding_dim),
#             nn.Dropout(dropout),
#             nn.Linear(n_embedding_dim, 2)
#         )                               
        
#         self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        
#         # Additional normalization layers
#         self.norm1 = nn.LayerNorm(n_embedding_dim)
#         self.norm2 = nn.LayerNorm(n_embedding_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        
#         self.relu = nn.ReLU(True)
#         self.softmaxd1 = nn.Softmax(dim=-1)

#     def forward(self, inputs):
#         # Enhanced feature processing - ensure float32 type
#         inputs = inputs.float()  # Convert to float32 to match model parameters
#         base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
#         base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
#         base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
#         base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        
#         # Split into short and long sequences for hierarchical processing
#         short_window_size = 16
#         if base_x.shape[0] > short_window_size:
#             short_x = base_x[-short_window_size:]
#             long_x = base_x[:-short_window_size]
            
#             # Process short sequence with standard encoder
#             pe_x = self.positional_encoding(short_x)
#             encoded_x = self.encoder(pe_x)
            
#             # Process long sequence with adaptive attention
#             if long_x.shape[0] > 0:
#                 attention_weights, adaptive_features, actual_spans = self.adaptive_span_attention(long_x, encoded_x)
                
#                 # Apply hierarchical encoding to long sequence
#                 hist_pe_x = self.positional_encoding(long_x)
#                 weighted_history = hist_pe_x * attention_weights.unsqueeze(-1)
#                 hierarchical_features = self.hierarchical_encoder(weighted_history, attention_weights)
                
#                 # Combine hierarchical features with current encoded features
#                 combined_features = torch.cat([hierarchical_features[-short_window_size:], encoded_x], dim=0)
#                 # Take mean to maintain original sequence length
#                 encoded_x = torch.mean(combined_features.view(2, short_window_size, -1, encoded_x.shape[-1]), dim=0)
#         else:
#             # For short sequences, use standard processing
#             pe_x = self.positional_encoding(base_x)
#             encoded_x = self.encoder(pe_x)
        
#         # Standard decoder processing
#         decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
#         decoded_x = self.decoder(decoder_token, encoded_x) 
        
#         # Add residual connection and normalization
#         decoded_x = self.norm1(decoded_x + self.dropout1(decoder_token))
        
#         decoded_x = decoded_x.permute([1, 0, 2])
        
#         anc_cls = self.classifier(decoded_x)
#         anc_reg = self.regressor(decoded_x)
        
#         return anc_cls, anc_reg

 
# class SuppressNet(torch.nn.Module):
#     def __init__(self, opt):
#         super(SuppressNet, self).__init__()
#         n_class=opt["num_of_class"]-1
#         n_seglen=opt["segment_size"]
#         n_embedding_dim=2*n_seglen
#         dropout=0.3
#         self.best_loss=1000000
#         self.best_map=0
#         # FC layers for the 2 streams
        
#         self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
#         self.mlp2 = nn.Linear(n_embedding_dim, 1)
#         self.norm = nn.InstanceNorm1d(n_class)
#         self.relu = nn.ReLU(True)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, inputs):
#         #inputs - batch x seq_len x class
        
#         base_x = inputs.permute([0,2,1])
#         base_x = self.norm(base_x)
#         x = self.relu(self.mlp1(base_x))
#         x = self.sigmoid(self.mlp2(x))
#         x = x.squeeze(-1)
        
#         return x

import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class AdaptiveSpanAttention(nn.Module):
    """
    Fast Adaptive Span Attention with efficient computation.
    """
    def __init__(self, embedding_dim, num_heads, dropout, max_span=64, min_span=8):
        super(AdaptiveSpanAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = min(num_heads, 4)  # Limit heads for speed
        self.max_span = min(max_span, 64)   # Limit span for speed
        self.min_span = min_span
        
        # Single efficient attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=self.num_heads, 
            dropout=dropout,
            batch_first=False
        )
        
        # Lightweight span predictor
        self.span_predictor = nn.Linear(embedding_dim, 1)
        
        # Simple gating
        self.gate = nn.Linear(embedding_dim, embedding_dim)
        
        # Temporal decay (fixed for speed)
        self.register_buffer('temporal_decay', torch.tensor(0.95))
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, long_x, encoded_x):
        """
        Fast forward pass with minimal computation.
        """
        batch_size = long_x.shape[1]
        full_seq_len = long_x.shape[0]
        
        # Simple span prediction (single value for all heads)
        context_rep = torch.mean(encoded_x, dim=0)  # [batch, dim]
        span_ratio = torch.sigmoid(self.span_predictor(context_rep))  # [batch, 1]
        
        # Compute adaptive span
        adaptive_span = self.min_span + (self.max_span - self.min_span) * span_ratio
        adaptive_span = adaptive_span.clamp(self.min_span, min(self.max_span, full_seq_len))
        avg_span = int(adaptive_span.mean().item())
        
        # Select relevant history (same span for all batches for efficiency)
        start_idx = max(0, full_seq_len - avg_span)
        relevant_history = long_x[start_idx:]  # [span_len, batch, dim]
        
        # Apply temporal decay
        seq_len = relevant_history.shape[0]
        positions = torch.arange(seq_len, device=long_x.device, dtype=torch.float)
        decay_weights = self.temporal_decay ** (seq_len - 1 - positions)
        decay_weights = decay_weights.unsqueeze(1).unsqueeze(2)  # [seq_len, 1, 1]
        
        # Apply decay to history
        weighted_history = relevant_history * decay_weights
        
        # Single attention computation
        attended_output, _ = self.attention(
            query=encoded_x,
            key=weighted_history,
            value=weighted_history
        )
        
        # Simple gating
        gate_weights = torch.sigmoid(self.gate(attended_output))
        adaptive_features = attended_output * gate_weights
        
        # Residual connection and normalization
        output = self.layer_norm(adaptive_features + encoded_x)
        
        # Return simplified outputs
        attention_weights = torch.ones(encoded_x.shape[0], batch_size, device=long_x.device)
        actual_spans = [avg_span]
        
        return attention_weights, output, actual_spans


class HierarchicalContextEncoder(nn.Module):
    """
    Fast hierarchical encoder with efficient multi-scale processing.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(HierarchicalContextEncoder, self).__init__()
        
        # Simplified encoders
        self.fine_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=min(num_heads, 4),  # Limit heads
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        self.coarse_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=min(num_heads // 2, 2),  # Even fewer heads
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        # Simple fusion
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Lightweight feature selection
        self.feature_selector = nn.Linear(embedding_dim * 2, 2)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, attention_mask):
        """
        Fast forward pass with minimal computation.
        """
        seq_len, batch_size, dim = features.shape
        
        # Fine scale processing
        fine_features = self.fine_encoder(features)
        
        # Coarse scale processing (only if beneficial)
        if seq_len >= 8:
            # Simple downsampling
            coarse_features = features.permute(1, 2, 0)  # [batch, dim, seq]
            coarse_features = F.avg_pool1d(coarse_features, kernel_size=2, stride=2, padding=0)
            coarse_features = coarse_features.permute(2, 0, 1)  # [seq//2, batch, dim]
            
            # Encode
            coarse_encoded = self.coarse_encoder(coarse_features)
            
            # Simple upsampling
            coarse_upsampled = coarse_encoded.permute(1, 2, 0)  # [batch, dim, seq//2]
            coarse_upsampled = F.interpolate(coarse_upsampled, size=seq_len, mode='nearest')
            coarse_upsampled = coarse_upsampled.permute(2, 0, 1)  # [seq, batch, dim]
        else:
            coarse_upsampled = fine_features
        
        # Simple feature selection
        combined = torch.cat([fine_features, coarse_upsampled], dim=-1)
        selection_weights = torch.softmax(self.feature_selector(combined), dim=-1)
        
        # Weighted combination
        output = (fine_features * selection_weights[:, :, 0:1] + 
                 coarse_upsampled * selection_weights[:, :, 1:2])
        
        # Simple fusion
        fused = self.fusion(combined)
        output = output + fused
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        
        # Enhanced feature reduction with normalization
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2)
        )
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        # Enhanced adaptive span attention
        self.adaptive_span_attention = AdaptiveSpanAttention(
            embedding_dim=n_embedding_dim,
            num_heads=min(4, n_enc_head),  # Limit heads for speed
            dropout=dropout,
            max_span=64,   # Reduced from 200
            min_span=8
        )
        
        # Enhanced hierarchical context encoder
        self.hierarchical_encoder = HierarchicalContextEncoder(
            embedding_dim=n_embedding_dim,
            num_heads=min(4, n_enc_head),  # Limit heads for speed
            dropout=dropout
        )
        
        # Main encoder with reduced layers (since we have hierarchical processing)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
                                     nhead=n_enc_head, 
                                     dropout=dropout, 
                                     activation='gelu'), 
            max(1, n_enc_layer - 1),  # Reduce by 1 since hierarchical encoder adds complexity
            nn.LayerNorm(n_embedding_dim)
        )
                                            
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                     nhead=n_dec_head, 
                                     dropout=dropout, 
                                     activation='gelu'), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        
        # Simplified context integration
        self.context_integration = nn.Sequential(
            nn.Linear(n_embedding_dim * 2, n_embedding_dim),
            nn.GELU()
        )
        
        # Enhanced classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, 2)
        )                               
        
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        
        # Additional normalization layers
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Enhanced feature processing
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        
        # Apply positional encoding
        pe_x = self.positional_encoding(base_x)
        
        # Conditional hierarchical processing (only for longer sequences)
        if pe_x.shape[0] > 8:
            attention_mask = torch.ones(pe_x.shape[0], pe_x.shape[1], device=pe_x.device)
            hierarchical_features = self.hierarchical_encoder(pe_x, attention_mask)
        else:
            hierarchical_features = pe_x
        
        # Standard encoder processing
        encoded_x = self.encoder(hierarchical_features)
        
        # Conditional adaptive attention (only for longer sequences)
        if pe_x.shape[0] > 16:
            _, adaptive_enhanced_features, _ = self.adaptive_span_attention(pe_x, encoded_x)
            # Integrate features
            integrated_features = self.context_integration(
                torch.cat([encoded_x, adaptive_enhanced_features], dim=-1)
            )
        else:
            integrated_features = encoded_x
        
        # Apply normalization
        integrated_features = self.norm1(integrated_features)
        
        # Decoder processing
        decoder_token = self.decoder_token.expand(-1, integrated_features.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, integrated_features) 
        
        # Add residual connection and normalization
        decoded_x = self.norm2(decoded_x + self.dropout1(decoder_token))
        
        decoded_x = decoded_x.permute([1, 0, 2])
        
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)
        
        return anc_cls, anc_reg

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
