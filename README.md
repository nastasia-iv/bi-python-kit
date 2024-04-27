This repository contains homework completed as part of the Python course during the annual retraining program at the Bioinformatics Institute (2023/2024).  
  
*Topics covered*: basic and advanced Python syntax, functions, working with files, OOP, API and web page parsing, code testing.  
  
Below is a brief description of the module:  
### 🌿 bi_python_kit.py  
  #### `filter_fastq`  
  Biopython-based FASTQ records filter working according to the specified criteria.  
  #### `BiologicalSequence`  
  OOP based class that allows to perform simple operations with biological sequences (DNA, RNA, amino acid sequences).  
  #### `telegram_logger`
  Decorator for logging function execution and sending logs to Telegram bot. Written without using libraries that automate the creation of a Telegram bot.
  #### `GenscanOutput`  
  Python API for http://hollywood.mit.edu/GENSCAN.html. Represents the output of the GENSCAN prediction.  
### 🌿 bio_file_processor.py  
  #### `convert_multiline_fasta_to_oneline`  
  Function for conversion multi-line FASTA sequences to single-line sequences.  
  #### `select_genes_from_gbk_to_fasta`  
  Function that select neighbor genes for the gene of interest from the GBK file and writes their protein sequences into FASTA format.  
  #### `OpenFasta`  
  Context manager for reading FASTA files.  
### 🌿 custom_random_forest.py  
  #### `RandomForestClassifierCustom`  
  Class for custom implementation of a Random Forest classifier. Uses thread parallelization to get results faster.  
### 🌿 test_script_pytest.py  
A set of tests using the paytest library that check the correct execution of some module functions.  
### 🌿 Showcases.ipynb
Contains selected examples of how this module works.
