# super-phage
- Embeddings of receptor-binding proteins show better performance compared to handcrafted genomic and protein sequence features.

- The transformer-based protein language model ProtT5 achieves the highest performance [^1^].

- For the feasibility study, we trained an MLP instead of a random forest from [^1^], resulting in an increase in weighted F1 scores from 62 to 78. (Is this SOTA for phage host interaction prediction?)

- We utilized optimal transport-based optimization for designing a cocktail of superphage.

Note: The code was developed for understanding and feasibility study of a student project and is research-oriented in nature. Please feel free to add improvements.

[1] Protein embeddings improve phage-host interaction prediction, June 2023, PLOS ONE, https://doi.org/10.1371/journal.pone.0289030
