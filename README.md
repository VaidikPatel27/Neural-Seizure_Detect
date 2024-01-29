# NeuroML_Detect
Machine Learning-Enabled Prediction of Brain Disease Identification from Electroencephalogram and Spectrogram Brain Activity.

### Details
The objective of this project is to identify and categorize seizures along with other forms of detrimental brain activity. Participants will be tasked with creating a model that is trained on electroencephalography (EEG) signals obtained from hospitalized patients in critical condition.

Currently, EEG monitoring relies solely on manual analysis by specialized neurologists. While invaluable, this labor-intensive process is a major bottleneck. Not only can it be time-consuming, but manual review of EEG recordings is also expensive, prone to fatigue-related errors, and suffers from reliability issues between different reviewers, even when those reviewers are experts.

The outcomes of efforts have the potential to significantly enhance the accuracy of classifying electroencephalography patterns. This breakthrough holds the promise of transformative advantages for neurocritical care, epilepsy management, and the field of drug development. Progress in this domain could empower healthcare professionals and brain researchers to swiftly identify seizures or other forms of brain damage, facilitating more prompt and precise treatment interventions.

#### Spectrogram example:

![spectrogram of brain activity_raw_continuous_30s](https://github.com/VaidikPatel27/NeuroML_Detect/assets/63740188/48ee2b86-4a7b-4dc1-a086-7b6dbd54955b)

In image all the different signals are different nodes or points on which we measure brain activity to get the EEG data.

<img width="1421" alt="brain nodes for EEG" src="https://github.com/VaidikPatel27/NeuroML_Detect/assets/63740188/a5dc01d3-b8e9-4646-bb3e-51c48042d78b">

### Data

Here we have data of different patient's EEG and Spectrogram files in the format of parquet files, with their patient ID. We have a train.csv file to get the time stemps with EEG and spectrogram ID's to fetch from the patient's parquet file of EEG and spectrogram.

#### Note:
Due to space limitations I can not upload entire dataset as it is 25GB+ dataset. Hence I uploaded 5 parquet files that I created from the entire dataset, named 'Neuro_Detect_train_{1 to 5}.parquet'. Before using it merge them. 
I also provided code to extract data from the files too, for referemce.






