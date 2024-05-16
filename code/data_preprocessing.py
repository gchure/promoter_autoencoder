#%%
import numpy as np 
import pandas as pd 
import tqdm
PROMOTER_WINDOW_LENGTH=45

# Load the ecocyc promoters and 
ecocyc_promoters = pd.read_csv('../data/ecocyc_transcription_units.txt', delimiter='\t')

# Drop NaN values 
ecocyc_promoters = ecocyc_promoters[(~ecocyc_promoters['Absolute-Plus-1-Position'].isnull()) & 
                                    (ecocyc_promoters['Absolute-Plus-1-Position'].str.contains('SUBSEQ') == False)]

# Cast the absolute +1 position to an integer
ecocyc_promoters['Absolute-Plus-1-Position'] = ecocyc_promoters['Absolute-Plus-1-Position'].astype(int)


# Add unique identifier for each promoter
ecocyc_promoters['promoter_id'] = np.arange(1, len(ecocyc_promoters)+1)

# Reformat the 'Genes' column to be a list of genes
ecocyc_promoters['Genes'] = ecocyc_promoters['Genes'].str.split('//')

# Set up a DNA mapper
int_mapper = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
rev_mapper = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
rev_comp = {0: 1, 1: 0, 2: 3, 3: 2}

# Load the FASTA file of the genome and encode it as an integer array
encoded_genome = []
with open('../data/escherichia_coli_K12_MG1655_genome.fasta', 'r') as f:
    genome = f.readlines()[1:]
    for line in genome:
        line = ''.join(line).replace('\n', '')
        for base in line:
            encoded_genome.append(int_mapper[base]) 
encoded_genome = np.array(encoded_genome)

#%%
#%%
proc_data = pd.DataFrame([])
for i in tqdm.tqdm(range(len(ecocyc_promoters))):

    # Get the promoter information
    promoter = ecocyc_promoters.iloc[i]
    plus1 = promoter['Absolute-Plus-1-Position']
    promoter_name = promoter['Promoter']
    strand = promoter['Transcription-Direction']
    promoter_id = promoter['promoter_id']
    genes = promoter['Genes']
    if len(genes) == 1:
        genes = genes[0]
    sigma_factors = promoter['Binds-Sigma-Factor']
    if sigma_factors is np.nan:
        sigma_factors = ['None']
    else:
        sigma_factors = sigma_factors.split(' // ')
        sigma_factors = [sigma.split('RNA polymerase sigma factor ')[1] for sigma in sigma_factors]
    if len(sigma_factors) == 1:
        sigma_factors = sigma_factors[0]

    # Get the promoter sequence
    if strand == '-':
        window = np.s_[plus1-1:plus1+PROMOTER_WINDOW_LENGTH]
        encoded_promoter = encoded_genome[window]
        encoded_promoter = np.array([rev_comp[base] for base in encoded_promoter[::-1]])
    else:
        window = np.s_[plus1-PROMOTER_WINDOW_LENGTH-1:plus1]
        encoded_promoter = np.array(encoded_genome[window])

    # Compute the reverse complement of the promoter
    seq = ''.join([rev_mapper[base] for base in encoded_promoter])

    # Set up the entry in the new dataframe
    _df = pd.DataFrame({'promoter_id': promoter_id,
                        'promoter_name': promoter_name,
                        'regulated_genes': [genes],
                        'sigma_factors': [sigma_factors],
                        'strand': strand,
                        'sequence_5-3': [seq]},
                        index=[0])
    proc_data = pd.concat([proc_data, _df], axis=0)
proc_data.to_csv('../data/promoter_sequences.csv', index=False)




