# Input
  # V3 nucleotide alignment
  # LANL metadata file  

# Output
  # Unlabeled Corpus
    # HGF Dataset with train/test/val
    # Ready for Masked Learning
  # Coreceptor
    # HGF Dataset with train/test/val
      # Tokenized with prot-bert
      # Labels in `Array2d` for multi-class prediction
    # Cleaned metadata for making tables
  # Body-site
    # HGF Dataset with train/test/val
      # Tokenized with prot-bert
      # Labels in `ClassLabel` for single-class prediction
    # Cleaned metadata for making tables