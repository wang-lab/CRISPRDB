To use the python version TSAM, the following packages must be installed:
1.xgboost
2.biopython
3.libsvm

The installation of these packages are as follows:

1. The installation of xgboost can be found at the website: https://xgboost.readthedocs.io/en/latest/build.html
It should be mentioned that the latest version package should be installed or errors may be occured such as cannot
use the parameter " 'objective': 'reg:tweedie' ". (we installed the xgboost-0.6a2 version)

2. The installation of the biopython can follow the decription at http://biopython.org/DIST/docs/install/Installation.html
The biopython-1.70 was installed when implementing the TSAM

3. libsvm can be downloaded from the website: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
Here, the python version should be installed. We installed the libsvm 3.22.

Inaddition, the following independent packages should be installed:
4. numpy
5. scipy
6. sklearn

The python2.7 is recommended for running these codes.
################################################################################################################################
################################################################################################################################
Part A. Prediction cutting efficiencies or classification of sgRNAs for a given gene' s patential sgRNAs
can be done via the following action:

   under windows os:
        open the command prompt
        cd xxx/TSAM/python_codes
        python TSAM_python.py filepath filetype species pretype sgtype phmmtype

  under linux os:
        open a command line tool:
        cd xxx/TSAM/python_codes
        python TSAM_python.py filepath filetype species pretype sgtype phmmtype

 ##############  illustration of the input parameters   ########################################################################
 ### filepath (string)--xxx.fa:
 ###                     the .fa sequence file that contain the gene sequence
 ### filetype (string)--annotated/genome/selfdefined:
 ###                     "annotated" means the gene sequences is downloaded from the ensembl database and contains
 ###                     detail gene annotation information such as exon, intron, 5'utr, 3'utr, protein and genome
 ###                     "genome" means the input is a complete genome DNA sequence (the header should contain keyword "GRC")
 ###                     "selfdefined" means a user provided gene sequences with the length more than 30
 ### species (int)---1/2/3:
 ###                     1 for human; 2 for mouse; 3 for zebrafish
 ###                    
 ### pretype (int)---1/0:
 ###                     1 for regression; 0 for classification
 ### sgtype (int)---1/0:
 ###                     1 for cut at exon only which means the sgRNAs cutting at the exons will be predicted;
 ###                     0 for all, where all the potential sgRNAs will be predicted
 ### phmmtype (int)---1/0:
 ###                     1 for using the pHMM feature; 0 for not use the pHMM feature

 ############## output file illustration #####################################################################################
 ### after finishing running the codes, one can find the result file in the .csv format at:
 ###        python_codes/predicted_scores/predict_results.csv

 ### There are four fields: spacer, xgboost_predict, libsvm_predict, TSAM_predict
 ###       spacer--- the potential spacer sequence (20nt)
 ###       xgboost_predict---the xgboost predicted cutting efficiency
 ###       libsvm_predict---the libsvm predicted citting efficiency
 ###       TSAM_predict---the final predicted score of the TSAM tool (usually, TSAM_predict=(xgboost_predict+libsvm_predict/2)
 ######################################################################################################################

 #### example codes(one can paste this to the command after adding the python_codes folder as the current working path):
    
    python TSAM_python.py ../test_files/test1.fa annotated 1 1 1 1

###############################################################################################################################
###############################################################################################################################
Part B.
To implement the cross-validation:
   step1: open the TSAM_python.py file
   step2: common the codes from line 2003 to line 2012:
   ***************************************************************************************************
   *    filepath = sys.argv[1]                                                                       *
   *    filepath=filepath.replace('\\\\', '/')                                                       *
   *    filepath=filepath.replace('\\', '/')                                                         *
   *    filepath=filepath.replace('//', '/')                                                         *
   *    filetype = sys.argv[2]                                                                       *
   *    species = sys.argv[3]                                                                        *
   *    pretype = int(sys.argv[4])　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　*
   *    sgtype = int(sys.argv[5])                                                                    *
   *    phmmtype = int(sys.argv[6])                                                                  *
   *    file_path, pre_scores=prediction_ce(filepath, filetype, species, pretype, sgtype, phmmtype)  *
   ***************************************************************************************************
   step3: uncommon the codes from line 1940 to line 1943:
   **************************************************************************************
   * #    data_index = 4                                                                *
   * #    pretype = 2                                                                   *
   * #    featype = 2                                                                   *
   * #    Data_index, Pretype, Featype = cross_validations(data_index, pretype, featype)*
   **************************************************************************************
   step4: save the file
   step5: open the command tool
   step6: cd xxx/TSAM/python_codes
   step7: python TSAM_python.py

#############################################################################################################################
#############################################################################################################################
Part C.
To implement the independent test:
   step1: open the TSAM_python.py file
   step2: common the codes from line 2003 to line 2012
   ***************************************************************************************************
   *    filepath = sys.argv[1]                                                                       *
   *    filepath=filepath.replace('\\\\', '/')                                                       *
   *    filepath=filepath.replace('\\', '/')                                                         *
   *    filepath=filepath.replace('//', '/')                                                         *
   *    filetype = sys.argv[2]                                                                       *
   *    species = sys.argv[3]                                                                        *
   *    pretype = int(sys.argv[4])　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　*
   *    sgtype = int(sys.argv[5])                                                                    *
   *    phmmtype = int(sys.argv[6])                                                                  *
   *    file_path, pre_scores=prediction_ce(filepath, filetype, species, pretype, sgtype, phmmtype)  *
   ***************************************************************************************************
   step3: uncommon the codes from line 1951 to line 1956
   ******************************************************************************
   *  #     data_index = 3                                                      *
   *  #     pretype = 1                                                         *
   *  #     featype = 2                                                         *
   *  #     phmmtype = 1                                                        *
   *  #     results = test_final_models(data_index, pretype, featype, phmmtype) *
   *  #     print results                                                       *
   ******************************************************************************
   step4: save the file
   step5: pen the command tool
   step6: cd xxx/TSAM/python_codes
   step7: python TSAM_python.py   
############################################################################################################################
############################################################################################################################
If you encount some problems to run the codes, please contact Hui Peng: Hui.Peng-2@student.uts.edu.au