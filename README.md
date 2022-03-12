# CRISPRDB
CRISPRDB is a tool for designing guide RNAs (gRNAs) for use in the CRISPR/Cas9 system for genome editing. You may use it freely in your research with appropriate acknowledgement. CRISPRDB is provided “as is” and the author is not responsible for consequences from the use of this program. CRISPRDB is distributed under the GNU General Public License v3.0 as published by the Free Software foundation.

## Requirements

* This package is supported for *Linux* operating systems. The package has been tested on the following systems:
```
   Linux: Ubuntu 18.04.4
   Linux: CentOS 7.1.1503
```
* Perl 5 interpreter or higher on a compatible Linux system is required.
   * [Installation instruction](https://learn.perl.org/installing/)
   * [Perl download](https://www.perl.org/get.html)
* The versions of Python3 packages which CRISPRDB used are, specifically:
```
   numpy: 1.21.5
   tensorflow: 2.4.1
   scipy: 1.7.3
   scikit-learn: 0.20.3
   xgboost: 0.80
   pandas: 1.3.5
   biopython: 1.79
   viennarna: 2.2.5
   joblib: 0.16.0
   keras: 2.2.4 (in order to run multiple public algorithms, we have a modified keras package for CRISPRDB saved in /crisprdb/ensemble/packages.zip)
   ray: 1.0.1
```
* The versions of Python2 packages which CRISPRDB used are, specifically:
```
   biopython: 1.73
	scipy: 1.1.0
	tensorflow: 1.4.1
	scikit-learn: 0.20.3
	xgboost: 0.82
	numpy: 1.14.5
```
  
   
## Installation and Usage

* Download the compressed file of CRISPRDB through 'Download zip' from Github.
* The easiest way to get the needed python3 prerequisites to run CRISPRDB is through conda. If you have conda installed already you can skip this step, otherwise go to https://docs.conda.io/en/latest/miniconda.html to learn how to install conda on your system. Once conda is correctly installed. You need to install the CRISPRDB requirements with
```
   conda create -y -c bioconda -c conda-forge -n crisprdb python=3 numpy=1.21.5 tensorflow=2.4.1 scipy=1.7.3 scikit-learn=0.20.3 xgboost=0.80 pandas=1.3.5 biopython=1.79 viennarna=2.2.5 joblib=0.16.0 
   pip3 install ray=1.0.1
```
* Install each python2 pachages using 'pip2 install package' command (replace 'package' with specific package names).
* Place the crisprdb.tar file anywhere in your Linux system and uncompress it.
* Extract all fills from /crisprdb/ensemble/packages.zip to /crisprdb/ensemble/packages.
* Copy your input FASTA file into the newly created crisprdb directory.
* Type 'perl crisprdb.pl' to run the program and view the help file.


## Command Line Parameters

* Direct sequence submission (-s or --sequence):
   
   This option allows the user to submit a single sequence directly for analysis, using the following command:
   ```
      perl crisprdb.pl –s <sequence> 
   ```
   This option is most useful for users who wish to determine the efficacy of a single gRNA. Any sequences submitted must be at least 31 bases long (including the NGG PAM region) and contain only A, T, U, C, or G. These rules also apply for any FASTA sequences that are submitted, which are covered in more detail in the next section.  
* FASTA file submission (-f or --file):
   
   This option allows the user to submit one or more sequences in a FASTA file, using the following command:
   ```
      perl crisprdb.pl –f myFastaFile.fasta 
   ```
   This should be provided in FASTA format. In a FASTA file, a definition line that begins with begins with ‘>’ is required for each DNA sequence. For example:
   ```
      >gi|4507798|ref|NM_000462.1| Homo sapiens ubiquitin protein ligase E3A (UBE3A), mRNA
      ATGGAGAAGCTGCACCAGTGTTATTGGAAATCAGGAGAACCTCAGTCTGACGACATTGAAGCTAGCCGA
      TGAAGCGAGCAGCTGCAAAGCATCTAATAGAACGCTACTACCACCAGTTAACTGAGGGCTGTGGAAATA
      AGCCTGCACGAATGAGTTTTGTGCTTCCTGTCCAACTTTTCTTCGTATGGATAATAATGCAGCAGCTAT
      TAAAGCCCTCGAGCTTTATAAGATTAATGCAAAACTCTGTGATCCTCATCCCTCCAAGAAAGGAGCAAG
      CGCAGCTTACCTTGAGAACTCGAAAGGTGCCCCCAACAACTCCTGCTCTGAGATAAAAATGAACAAGAA
      AGG
   ```
   Submitted sequences must be between 31 and 100,000 nt in length and contain A, T, U, C, or G. 
* Sample file submission:
   Three sample files are also provided: one containing a single short sequence (31 nt), one with a single long sequence (8,322 nt), and one with 3 sequences of 300, 600, and 300 bases long. This option allows the user to try one of the three previously mentioned sample files, using the following command:
   ```
      perl crisprdb.pl –e <short|long|multiple>
   ```
   One of the three options shown will call the respective sample file.
   
## Sample expected outputs
After running the above commands, the results are made available in a tab-delimited text file saved to the crisprdb/result folder.  The first line of the file contains the column headers.  Each line in the result file shows (in order) the sequence identifier, the efficacy score of the gRNA, the gRNA sequence, the orientation of the gRNA, and the location of the gRNA in the target sequence. The gRNAs are sorted by score, with higher scores indicating greater effectiveness. All gRNAs are 20 bases long. An example result table from predicting the sample file of the single long sequence (8322 nt) would be:

![sample_output2](https://github.com/wang-lab/CRISPRDB/blob/main/example_result_table.jpg)
