# BAM — Behance Appreciation Model
BAM aims to uncover what makes a Behance project successful by identifying the visual, textual, and creator-level features that drive total “appreciates.” The project builds a full end-to-end pipeline: it ingests raw Behance JSON files, extracts image and metadata features, cleans and merges millions of records, performs dimensionality reduction using PCA, and runs regression analysis to understand which attributes best explain popularity. All outputs—clean datasets, PCA-enhanced panels, and regression visualisations—are organized for reproducibility, with automated plotting and tidy Parquet files so both analysts and non-technical users can explore how artistic choices influence audience engagement.

## Folder Structure

BAM_DS/
- raw_data/
   - Behance_appreciate_1M/         # Original 1M Behance JSON files
   - Behance_image_features.b/                # Raw image feature binaries
   - Behance_item_to_owners/                # Creator → project mapping files
 
- src/
  - extract_appreciates.py         # Extracts appreciates + metadata from raw JSON
  - extract_features.py            # Reads image feature binaries
  - merge_owner_mappings.py        # Connects projects to owners
  - merge_all.py                   # Combines everything & creates final_dataset.parquet
  - pca_analysis_clean.py                     # Applies PCA and outputs final_dataset_with_pca.parquet
  - regressions.py       # Runs regressions and saves visualizations
    
- data_processed/
  - merged_raw_dataset.parquet
  - final_dataset.parquet
  - final_dataset_with_pca.parquet

- visualisations/
  - *.png files                          # Automatically generated plots

## Execution Order

1. extract_appreciates.py
Extracts appreciates + metadata from raw Behance JSON.

2. extract_features.py
Loads image feature binaries and converts them to structured CSV/Parquet.

3. merge_owner_mappings.py
Connects each project to its creator.

4. merge_all.py
Merges all components into final_dataset.parquet.

5. pca_analysis_clean.py
Performs PCA and outputs final_dataset_with_pca.parquet.

6. regressions.py
Runs regression models and generates all visualisations.
