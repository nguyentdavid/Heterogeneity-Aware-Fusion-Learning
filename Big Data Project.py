# Importing necessary libraries for data manipulation, deep learning, and analysis.
import numpy as np # Importing numpy for numerical operations.
import pandas as pd # Importing pandas for data manipulation and analysis.
import torch # Importing torch for building and training neural networks.
import torch.nn as nn # Importing torch.nn for neural network modules.
import torch.optim as optim # Importing torch.optim for optimization algorithms.
from torch.utils.data import Dataset, DataLoader # Importing Dataset and DataLoader for handling data in PyTorch.
from sklearn.model_selection import train_test_split # Importing train_test_split for splitting data into training and testing sets.
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score # Importing metrics for evaluating model performance.
from sklearn.preprocessing import StandardScaler # Importing StandardScaler for standardizing features.
from sklearn.cluster import KMeans # Importing KMeans for clustering data.
# Mock lifelines for survival analysis placeholder
# from lifelines import KaplanMeierFitter # Importing KaplanMeierFitter for survival curve estimation.
# from lifelines.statistics import logrank_test # Importing logrank_test for comparing survival curves.
import matplotlib.pyplot as plt # Importing matplotlib for plotting.
import seaborn as sns # Importing seaborn for enhanced visualizations.
from umap import UMAP # Importing UMAP for dimensionality reduction and visualization.

# --- 1. DATA LOADING AND PREPROCESSING ---

# This section will contain functions to load and preprocess the four data modalities
# from the TCGA-LUAD dataset.

# Defining a function to load and preprocess genomics data.
def load_genomics_data(path="mock_data/genomics.csv"):
    """Loads and preprocesses genomics (somatic mutation) data."""
    print("Loading genomics data...") # Printing a message indicating the start of genomics data loading.
    # In a real scenario, this would load a file of somatic mutations.
    # We'll create a mock dataframe.
    # ~20,000 genes, ~585 patients
    patients = [f"TCGA-05-{i:04d}" for i in range(585)] # Generating a list of mock patient IDs.
    genes = [f"GENE_{j}" for j in range(20000)] # Generating a list of mock gene IDs.
    data = np.random.randint(0, 2, size=(len(patients), len(genes))) # Creating a mock data array with random integers.
    df = pd.DataFrame(data, index=patients, columns=genes) # Creating a pandas DataFrame from the mock data.
    df.index.name = "case_id" # Setting the index name of the DataFrame.
    print("Genomics data loaded.") # Printing a message indicating the completion of genomics data loading.
    return df # Returning the mock genomics DataFrame.

# Defining a function to load and preprocess transcriptomics data.
def load_transcriptomics_data(path="mock_data/transcriptomics.csv"):
    """Loads and preprocesses transcriptomics (gene expression) data."""
    print("Loading transcriptomics data...") # Printing a message indicating the start of transcriptomics data loading.
    # Mock RNA-Seq data
    patients = [f"TCGA-05-{i:04d}" for i in range(585)] # Generating a list of mock patient IDs.
    genes = [f"GENE_{j}" for j in range(20000)] # Generating a list of mock gene IDs.
    data = np.random.rand(len(patients), len(genes)) * 500 # Creating a mock data array with random floating-point numbers.
    df = pd.DataFrame(data, index=patients, columns=genes) # Creating a pandas DataFrame from the mock data.
    df.index.name = "case_id" # Setting the index name of the DataFrame.
    scaler = StandardScaler() # Initializing a StandardScaler object.
    df[:] = scaler.fit_transform(df) # Scaling the transcriptomics data.
    print("Transcriptomics data loaded and scaled.") # Printing a message indicating the completion of transcriptomics data loading and scaling.
    return df # Returning the scaled mock transcriptomics DataFrame.

# Defining a function to load pre-extracted imaging features.
def load_imaging_data(path="mock_data/imaging_features.csv"):
    """
    Loads pre-extracted imaging features.
    In a real-world scenario, this function would process whole-slide images,
    which is computationally expensive. For this script, we assume features
    have been pre-extracted using a pre-trained CNN (e.g., ResNet).
    """
    print("Loading imaging features...") # Printing a message indicating the start of imaging features loading.
    patients = [f"TCGA-05-{i:04d}" for i in range(585)] # Generating a list of mock patient IDs.
    # Example: 512 features from a CNN
    num_features = 512 # Defining the number of mock imaging features.
    features = np.random.randn(len(patients), num_features) # Creating a mock data array with random normal distributed numbers.
    df = pd.DataFrame(features, index=patients, columns=[f"IMG_F_{i}" for i in range(num_features)]) # Creating a pandas DataFrame from the mock data.
    df.index.name = "case_id" # Setting the index name of the DataFrame.
    print("Imaging features loaded.") # Printing a message indicating the completion of imaging features loading.
    return df # Returning the mock imaging features DataFrame.

# Defining a function to load and preprocess clinical data.
def load_clinical_data(path="mock_data/clinical.csv"):
    """Loads and preprocesses clinical data."""
    print("Loading clinical data...") # Printing a message indicating the start of clinical data loading.
    patients = [f"TCGA-05-{i:04d}" for i in range(585)] # Generating a list of mock patient IDs.
    data = {
        "age_at_diagnosis": np.random.randint(40, 80, size=len(patients)), # Creating mock age data.
        "vital_status": np.random.choice(["Alive", "Dead"], size=len(patients), p=[0.6, 0.4]), # Creating mock vital status data.
        "days_to_death": np.random.randint(30, 2000, size=len(patients)), # Creating mock days to death data.
        "pathologic_stage": np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], size=len(patients)) # Creating mock pathologic stage data.
    }
    df = pd.DataFrame(data, index=patients) # Creating a pandas DataFrame from the mock data.
    df.index.name = "case_id" # Setting the index name of the DataFrame.

    # Target for Task 1: Early Disease Detection (mortality within one year)
    df['early_mortality'] = ((df['vital_status'] == 'Dead') & (df['days_to_death'] <= 365)).astype(int) # Creating the early mortality target variable.

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['pathologic_stage'], drop_first=True) # Performing one-hot encoding on the 'pathologic_stage' column.

    # Scale continuous features
    scaler = StandardScaler() # Initializing a StandardScaler object.
    df[['age_at_diagnosis', 'days_to_death']] = scaler.fit_transform(df[['age_at_diagnosis', 'days_to_death']]) # Scaling continuous clinical features.
    print("Clinical data loaded and processed.") # Printing a message indicating the completion of clinical data loading and processing.
    return df # Returning the processed mock clinical DataFrame.

# Defining a function to align and merge data from different modalities.
def align_and_merge_data(genomics, transcriptomics, imaging, clinical):
    """Aligns all data modalities by patient ID."""
    print("Aligning data...") # Printing a message indicating the start of data alignment.
    # Find common patients across all modalities with complete data
    common_patients = list(set(genomics.index) & set(transcriptomics.index) & set(imaging.index) & set(clinical.index)) # Finding the intersection of patient IDs across all DataFrames.
    print(f"Found {len(common_patients)} patients with all four data modalities.") # Printing the number of common patients found.

    genomics = genomics.loc[common_patients] # Subsetting the genomics DataFrame to include only common patients.
    transcriptomics = transcriptomics.loc[common_patients] # Subsetting the transcriptomics DataFrame to include only common patients.
    imaging = imaging.loc[common_patients] # Subsetting the imaging DataFrame to include only common patients.
    clinical = clinical.loc[common_patients] # Subsetting the clinical DataFrame to include only common patients.

    labels = clinical['early_mortality'] # Extracting the 'early_mortality' column as labels.
    # Drop non-feature columns from clinical data
    clinical_features = clinical.drop(columns=['vital_status', 'days_to_death', 'early_mortality']) # Dropping non-feature columns from the clinical DataFrame.

    return {
        'genomics': genomics, # Returning the aligned genomics DataFrame.
        'transcriptomics': transcriptomics, # Returning the aligned transcriptomics DataFrame.
        'imaging': imaging, # Returning the aligned imaging DataFrame.
        'clinical': clinical_features # Returning the aligned clinical features DataFrame.
    }, labels # Returning the dictionary of aligned data and the labels.

# --- 2. PyTorch Dataset and DataLoader ---

# Defining a custom PyTorch Dataset for multimodal data.
class MultimodalDataset(Dataset):
    """PyTorch dataset for multimodal data."""
    def __init__(self, data, labels): # Defining the constructor for the dataset.
        self.data = data # Storing the input data dictionary.
        self.labels = labels # Storing the labels Series.
        self.patient_ids = list(self.labels.index) # Extracting patient IDs from the labels index.

    def __len__(self): # Defining the method to return the size of the dataset.
        return len(self.patient_ids) # Returning the number of patient IDs.

    def __getitem__(self, idx): # Defining the method to get a sample from the dataset by index.
        patient_id = self.patient_ids[idx] # Getting the patient ID for the given index.
        genomics_data = torch.tensor(self.data['genomics'].loc[patient_id].values, dtype=torch.float32) # Converting genomics data to a PyTorch tensor.
        transcriptomics_data = torch.tensor(self.data['transcriptomics'].loc[patient_id].values, dtype=torch.float32) # Converting transcriptomics data to a PyTorch tensor.
        imaging_data = torch.tensor(self.data['imaging'].loc[patient_id].values, dtype=torch.float32) # Converting imaging data to a PyTorch tensor.
        clinical_data = torch.tensor(self.data['clinical'].loc[patient_id].values, dtype=torch.float32) # Converting clinical data to a PyTorch tensor.
        label = torch.tensor(self.labels.loc[patient_id], dtype=torch.float32) # Converting the label to a PyTorch tensor.

        return {
            'genomics': genomics_data, # Returning the genomics tensor.
            'transcriptomics': transcriptomics_data, # Returning the transcriptomics tensor.
            'imaging': imaging_data, # Returning the imaging tensor.
            'clinical': clinical_data, # Returning the clinical tensor.
            'label': label.unsqueeze(0) # Returning the label tensor, unsqueezed to have a dimension for the batch.
        }


# --- 3. MODEL ARCHITECTURES ---

# Unimodal Models (Baselines)
# Defining a simple MLP model for a single modality.
class UnimodalMLP(nn.Module):
    """A simple MLP for any single modality."""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1): # Defining the constructor for the MLP.
        super(UnimodalMLP, self).__init__() # Calling the constructor of the parent class.
        self.network = nn.Sequential( # Defining a sequential network.
            nn.Linear(input_dim, hidden_dim), # Adding a linear layer.
            nn.ReLU(), # Adding a ReLU activation function.
            nn.Dropout(0.5), # Adding a dropout layer.
            nn.Linear(hidden_dim, hidden_dim // 2), # Adding another linear layer.
            nn.ReLU(), # Adding a ReLU activation function.
            nn.Dropout(0.5), # Adding a dropout layer.
            nn.Linear(hidden_dim // 2, output_dim), # Adding the output linear layer.
            nn.Sigmoid() # Adding a Sigmoid activation function for binary classification.
        )

    def forward(self, x): # Defining the forward pass of the MLP.
        return self.network(x) # Passing the input through the sequential network.

# Simple Concatenation Fusion Model (Baseline)
# Defining a simple fusion model that concatenates features.
class SimpleFusion(nn.Module):
    """Concatenates all modality features and feeds them to an MLP."""
    def __init__(self, input_dims, hidden_dim=256, output_dim=1): # Defining the constructor for the fusion model.
        super(SimpleFusion, self).__init__() # Calling the constructor of the parent class.
        total_input_dim = sum(input_dims.values()) # Calculating the total input dimension by summing up dimensions of all modalities.
        self.fusion_network = nn.Sequential( # Defining a sequential fusion network.
            nn.Linear(total_input_dim, hidden_dim), # Adding a linear layer.
            nn.ReLU(), # Adding a ReLU activation function.
            nn.BatchNorm1d(hidden_dim), # Adding a batch normalization layer.
            nn.Dropout(0.5), # Adding a dropout layer.
            nn.Linear(hidden_dim, hidden_dim // 2), # Adding another linear layer.
            nn.ReLU(), # Adding a ReLU activation function.
            nn.BatchNorm1d(hidden_dim // 2), # Adding another batch normalization layer.
            nn.Dropout(0.5), # Adding a dropout layer.
            nn.Linear(hidden_dim // 2, output_dim), # Adding the output linear layer.
            nn.Sigmoid() # Adding a Sigmoid activation function.
        )

    def forward(self, x): # Defining the forward pass of the fusion model.
        # x is a dictionary of tensors
        concatenated = torch.cat(list(x.values()), dim=1) # Concatenating the tensors from all modalities along dimension 1.
        return self.fusion_network(concatenated) # Passing the concatenated tensor through the fusion network.


# Heterogeneity-Aware Fusion Model (Proposed)
# Defining a modality-specific encoder.
class ModalityEncoder(nn.Module):
    """A dedicated encoder for each modality."""
    def __init__(self, input_dim, output_dim=128): # Defining the constructor for the encoder.
        super(ModalityEncoder, self).__init__() # Calling the constructor of the parent class.
        self.encoder = nn.Sequential( # Defining a sequential encoder network.
            nn.Linear(input_dim, input_dim // 2), # Adding a linear layer.
            nn.ReLU(), # Adding a ReLU activation function.
            nn.Linear(input_dim // 2, output_dim) # Adding the output linear layer.
        )

    def forward(self, x): # Defining the forward pass of the encoder.
        return self.encoder(x) # Passing the input through the encoder network.

# Defining a cross-modal attention mechanism.
class CrossModalAttention(nn.Module):
    """Transformer-based cross-modal attention mechanism."""
    def __init__(self, embed_dim=128, num_heads=4): # Defining the constructor for the attention mechanism.
        super(CrossModalAttention, self).__init__() # Calling the constructor of the parent class.
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # Initializing a MultiheadAttention module.

    def forward(self, query, key, value): # Defining the forward pass of the attention mechanism.
        # Each input is (batch_size, sequence_len, embed_dim)
        # For us, sequence_len will be 1 as we attend to the whole feature vector
        attn_output, _ = self.attention(query, key, value) # Applying multi-head attention.
        return attn_output # Returning the output of the attention mechanism.

# Defining the proposed heterogeneity-aware fusion model.
class ProposedFusionModel(nn.Module):
    """The proposed heterogeneity-aware fusion model."""
    def __init__(self, input_dims, embed_dim=128, num_heads=4, hidden_dim=256, output_dim=1): # Defining the constructor for the proposed model.
        super(ProposedFusionModel, self).__init__() # Calling the constructor of the parent class.
        # 1. Modality-Specific Encoders
        self.gen_encoder = ModalityEncoder(input_dims['genomics'], embed_dim) # Initializing the genomics encoder.
        self.trn_encoder = ModalityEncoder(input_dims['transcriptomics'], embed_dim) # Initializing the transcriptomics encoder.
        self.img_encoder = ModalityEncoder(input_dims['imaging'], embed_dim) # Initializing the imaging encoder.
        self.cli_encoder = ModalityEncoder(input_dims['clinical'], embed_dim) # Initializing the clinical encoder.

        # 2. Cross-Modal Attention
        self.attention = CrossModalAttention(embed_dim, num_heads) # Initializing the cross-modal attention module.

        # 3. Classifier for Fused Representation
        self.classifier = nn.Sequential( # Defining a sequential classifier network.
            nn.Linear(embed_dim * 4, hidden_dim), # Adding a linear layer, input dimension is the sum of embedded dimensions.
            nn.ReLU(), # Adding a ReLU activation function.
            nn.Dropout(0.5), # Adding a dropout layer.
            nn.Linear(hidden_dim, output_dim), # Adding the output linear layer.
            nn.Sigmoid() # Adding a Sigmoid activation function.
        )

    def forward(self, x): # Defining the forward pass of the proposed model.
        # x is a dictionary of tensors
        gen_embed = self.gen_encoder(x['genomics']) # Encoding the genomics data.
        trn_embed = self.trn_encoder(x['transcriptomics']) # Encoding the transcriptomics data.
        img_embed = self.img_encoder(x['imaging']) # Encoding the imaging data.
        cli_embed = self.cli_encoder(x['clinical']) # Encoding the clinical data.

        # Reshape for attention: (batch_size, 1, embed_dim)
        gen_embed_r = gen_embed.unsqueeze(1) # Reshaping the genomics embedding for attention.
        trn_embed_r = trn_embed.unsqueeze(1) # Reshaping the transcriptomics embedding for attention.
        img_embed_r = img_embed.unsqueeze(1) # Reshaping the imaging embedding for attention.
        cli_embed_r = cli_embed.unsqueeze(1) # Reshaping the clinical embedding for attention.

        # Apply attention (example: using genomics as query)
        # A more complex setup could involve self-attention or a different query
        gen_attended = self.attention(gen_embed_r, torch.cat([trn_embed_r, img_embed_r, cli_embed_r], dim=1), torch.cat([trn_embed_r, img_embed_r, cli_embed_r], dim=1)) # Applying attention with genomics as query.
        trn_attended = self.attention(trn_embed_r, torch.cat([gen_embed_r, img_embed_r, cli_embed_r], dim=1), torch.cat([gen_embed_r, img_embed_r, cli_embed_r], dim=1)) # Applying attention with transcriptomics as query.
        img_attended = self.attention(img_embed_r, torch.cat([gen_embed_r, trn_embed_r, cli_embed_r], dim=1), torch.cat([gen_embed_r, trn_embed_r, cli_embed_r], dim=1)) # Applying attention with imaging as query.
        cli_attended = self.attention(cli_embed_r, torch.cat([gen_embed_r, trn_embed_r, img_embed_r], dim=1), torch.cat([gen_embed_r, trn_embed_r, img_embed_r], dim=1)) # Applying attention with clinical as query.

        # Concatenate attended features
        fused_representation = torch.cat([gen_attended.squeeze(1), trn_attended.squeeze(1), img_attended.squeeze(1), cli_attended.squeeze(1)], dim=1) # Concatenating the attended features.

        # Get final prediction
        prediction = self.classifier(fused_representation) # Passing the fused representation through the classifier.
        return prediction, fused_representation # Returning the prediction and the fused representation.


# --- 4. TRAINING AND EVALUATION ---
# Defining a function for training the model for one epoch.
def train_model(model, dataloader, criterion, optimizer, device):
    """Training loop for one epoch."""
    model.train() # Setting the model to training mode.
    running_loss = 0.0 # Initializing the running loss.
    for batch in dataloader: # Iterating through the data loader.
        # Move data to device
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'} # Moving input data to the specified device.
        labels = batch['label'].to(device) # Moving labels to the specified device.

        optimizer.zero_grad() # Zeroing the gradients.

        # Forward pass
        if isinstance(model, ProposedFusionModel): # Checking if the model is the ProposedFusionModel.
            outputs, _ = model(inputs) # Performing the forward pass and getting outputs and embeddings.
        elif isinstance(model, UnimodalMLP): # Checking if the model is a UnimodalMLP.
             # This assumes we will adapt the dataloader to return single modalities
             # For simplicity, we'll focus on multimodal models for now.
             # This part needs adjustment for unimodal training.
            pass # Placeholder
        else:
            # Handle SimpleFusion
            outputs = model(inputs) # Performing the forward pass for SimpleFusion.

        loss = criterion(outputs, labels) # Calculating the loss.
        loss.backward() # Performing backpropagation.
        optimizer.step() # Updating model weights.
        running_loss += loss.item() # Accumulating the running loss.

    return running_loss / len(dataloader) # Returning the average loss for the epoch.


# Defining a function for evaluating the model.
def evaluate_model(model, dataloader, device):
    """Evaluation loop."""
    model.eval() # Setting the model to evaluation mode.
    all_labels = [] # Initializing a list to store all true labels.
    all_preds = [] # Initializing a list to store all predicted labels.
    with torch.no_grad(): # Disabling gradient calculation.
        for batch in dataloader: # Iterating through the data loader.
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'} # Moving input data to the specified device.
            labels = batch['label'].to(device) # Moving labels to the specified device.

            if isinstance(model, ProposedFusionModel): # Checking if the model is the ProposedFusionModel.
                outputs, _ = model(inputs) # Performing the forward pass and getting outputs.
            else:
                outputs = model(inputs) # Performing the forward pass for other models.

            preds = (outputs > 0.5).float() # Converting model outputs to binary predictions.
            all_labels.extend(labels.cpu().numpy()) # Appending true labels to the list.
            all_preds.extend(preds.cpu().numpy()) # Appending predicted labels to the list.

    accuracy = accuracy_score(all_labels, all_preds) # Calculating accuracy.
    f1 = f1_score(all_labels, all_preds) # Calculating F1-score.
    auroc = roc_auc_score(all_labels, all_preds) # Calculating AUROC.
    return accuracy, f1, auroc # Returning the calculated metrics.


# --- 5. DISEASE SUBTYPING ---
# Defining a function to perform disease subtyping.
def perform_subtyping(model, dataloader, device, n_clusters=3):
    """
    Performs disease subtyping using the fused representations from the model.
    """
    model.eval() # Setting the model to evaluation mode.
    all_embeddings = [] # Initializing a list to store all embeddings.
    all_patient_ids = [] # To map back clusters to patients # Initializing a list to store patient IDs.

    with torch.no_grad(): # Disabling gradient calculation.
        for batch in dataloader: # Iterating through the data loader.
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'} # Moving input data to the specified device.
            _, embeddings = model(inputs) # Performing the forward pass to get embeddings.
            all_embeddings.append(embeddings.cpu().numpy()) # Appending embeddings to the list.
            # This part needs modification to get patient IDs from the dataloader
            # For now, we assume we can get them.

    all_embeddings = np.concatenate(all_embeddings, axis=0) # Concatenating all embeddings.

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) # Initializing a KMeans clustering object.
    clusters = kmeans.fit_predict(all_embeddings) # Performing KMeans clustering on the embeddings.

    # Visualization
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42) # Initializing a UMAP reducer object.
    umap_embedding = umap_reducer.fit_transform(all_embeddings) # Reducing the dimensionality of embeddings using UMAP.

    plt.figure(figsize=(10, 8)) # Creating a new figure for plotting.
    sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], hue=clusters, palette=sns.color_palette("hsv", n_clusters)) # Creating a scatter plot of UMAP embeddings, colored by cluster.
    plt.title('UMAP projection of patient clusters') # Setting the title of the plot.
    plt.xlabel('UMAP 1') # Setting the x-axis label.
    plt.ylabel('UMAP 2') # Setting the y-axis label.
    plt.legend() # Displaying the legend.
    plt.savefig("disease_subtypes_umap.png") # Saving the plot to a file.
    plt.show() # Displaying the plot.

    # Survival Analysis (Placeholder)
    print("\n--- Survival Analysis (Placeholder) ---") # Printing a placeholder message for survival analysis.
    print("This would involve using the 'lifelines' library to perform a log-rank test on the identified clusters.") # Explaining the placeholder.
    # Example:
    # clinical_data_with_clusters = clinical_data.copy() # Creating a copy of clinical data.
    # clinical_data_with_clusters['cluster'] = clusters # Adding cluster assignments to the clinical data.
    # kmf = KaplanMeierFitter() # Initializing a KaplanMeierFitter.
    # for i in range(n_clusters): # Iterating through the clusters.
    #     cluster_data = clinical_data_with_clusters[clinical_data_with_clusters['cluster'] == i] # Subsetting data for the current cluster.
    #     kmf.fit(cluster_data['days_to_death'], cluster_data['vital_status'], label=f'Cluster {i}') # Fitting the Kaplan-Meier model for the cluster.
    #     kmf.plot_survival_function() # Plotting the survival function for the cluster.
    # plt.title('Survival Curves by Patient Subtype') # Setting the title of the survival plot.
    # plt.show() # Displaying the survival plot.


# --- MAIN EXECUTION ---
if __name__ == '__main__': # Checking if the script is being run directly.
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting the device to GPU if available, otherwise CPU.
    print(f"Using device: {device}") # Printing the device being used.

    # 1. Load Data
    gen_data = load_genomics_data() # Loading genomics data.
    trn_data = load_transcriptomics_data() # Loading transcriptomics data.
    img_data = load_imaging_data() # Loading imaging data.
    cli_data = load_clinical_data() # Loading clinical data.

    # 2. Align Data
    all_data, labels = align_and_merge_data(gen_data, trn_data, img_data, cli_data) # Aligning and merging the data modalities.
    patient_ids = labels.index # Getting the patient IDs from the labels index.

    # 3. Data Splits
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42, stratify=labels) # Splitting data into training and testing sets.
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=42) # 0.1 of original # Splitting the training set into training and validation sets.

    # Defining a helper function to subset data.
    def subset_data(data, ids):
        sub_data = {} # Initializing an empty dictionary for subset data.
        for modality, df in data.items(): # Iterating through the data modalities.
            sub_data[modality] = df.loc[ids] # Subsetting the DataFrame for the current modality using the provided IDs.
        return sub_data # Returning the dictionary of subset data.

    train_data = subset_data(all_data, train_ids) # Creating the training data subset.
    val_data = subset_data(all_data, val_ids) # Creating the validation data subset.
    test_data = subset_data(all_data, test_ids) # Creating the test data subset.

    train_labels = labels.loc[train_ids] # Creating the training labels subset.
    val_labels = labels.loc[val_ids] # Creating the validation labels subset.
    test_labels = labels.loc[test_ids] # Creating the test labels subset.

    # 4. Create Datasets and DataLoaders
    train_dataset = MultimodalDataset(train_data, train_labels) # Creating a MultimodalDataset for training data.
    val_dataset = MultimodalDataset(val_data, val_labels) # Creating a MultimodalDataset for validation data.
    test_dataset = MultimodalDataset(test_data, test_labels) # Creating a MultimodalDataset for test data.

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Creating a DataLoader for the training dataset.
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # Creating a DataLoader for the validation dataset.
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Creating a DataLoader for the test dataset.

    # 5. Initialize and Train Model
    input_dimensions = {
        'genomics': all_data['genomics'].shape[1], # Getting the input dimension for genomics data.
        'transcriptomics': all_data['transcriptomics'].shape[1], # Getting the input dimension for transcriptomics data.
        'imaging': all_data['imaging'].shape[1], # Getting the input dimension for imaging data.
        'clinical': all_data['clinical'].shape[1] # Getting the input dimension for clinical data.
    }

    # --- Training the Proposed Model ---
    print("\n--- Training Proposed Heterogeneity-Aware Fusion Model ---") # Printing a message indicating the start of training.
    proposed_model = ProposedFusionModel(input_dimensions).to(device) # Initializing the ProposedFusionModel and moving it to the device.
    optimizer = optim.Adam(proposed_model.parameters(), lr=0.001) # Initializing the Adam optimizer.
    criterion = nn.BCELoss() # Initializing the Binary Cross-Entropy Loss criterion.

    num_epochs = 10 # Using a small number for demonstration # Defining the number of training epochs.
    for epoch in range(num_epochs): # Looping through the epochs.
        train_loss = train_model(proposed_model, train_loader, criterion, optimizer, device) # Training the model for one epoch.
        val_acc, val_f1, val_auroc = evaluate_model(proposed_model, val_loader, device) # Evaluating the model on the validation set.
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}") # Printing the training and validation metrics.

    # 6. Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---") # Printing a message indicating the start of test evaluation.
    test_acc, test_f1, test_auroc = evaluate_model(proposed_model, test_loader, device) # Evaluating the model on the test set.
    print(f"Test Accuracy: {test_acc:.4f}") # Printing the test accuracy.
    print(f"Test F1-Score: {test_f1:.4f}") # Printing the test F1-score.
    print(f"Test AUROC: {test_auroc:.4f}") # Printing the test AUROC.

    # 7. Perform Disease Subtyping
    print("\n--- Performing Disease Subtyping ---") # Printing a message indicating the start of disease subtyping.
    # Using the full dataset loader for subtyping
    full_dataset = MultimodalDataset(all_data, labels) # Creating a MultimodalDataset using the full dataset.
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False) # Creating a DataLoader for the full dataset.
    perform_subtyping(proposed_model, full_loader, device) # Performing disease subtyping.