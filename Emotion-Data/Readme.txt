Emotion-labeled dataset

------------------------------------------------------------------------------------------

Introduction
The dataset consists of 173 blog posts (drawn from the Web) containing a total of 15205 sentences. These sentences were labeled with emotion related by four judges such that each sentence was sunjected to independent evaluations. The judges were asked to indicate the emotion category (one of the pre-specified categories) and the emotion intensity (high, medium and low) conveyed by each sentence. They were also asked to mark in each sentence the words or strings of words that indicate emotion in the sentence.

Details about the annotation scheme and the agreement between the judges can be found in:
S. Aman and S. Szpakowicz (2007). "Identifying Expressions of Emotion in Text". In Proceedings of the 10th International Conference on Text, Speech, and Dialogue, Plzen, Czech Republic, LNCS, Springer.

------------------------------------------------------------------------------------------

Dataset Contents

1. Unannotated Data
2. Annotated Data
3. Benchmark

----------------------------------------------------------------------------------------------

1. Unannotated Data

* blogdata.txt
  contains the blog data with individual posts and sentences demarcated

2. Annotated Data

* AnnotSet1.txt
  contains the first set of annotations (performed by annotator A)
  + Format: <id> <emotion category label> <emotion intensity label> <emotion indicators>
     if the emotion category label is "ne" (no emotion), there is no emotion intensity label and no emotion indicators
  
* AnnotSet2.txt
  contains the second set of annotations (performed by annotators B, C, and D)
  + Format: <id> <emotion category label> <emotion intensity label> <emotion indicators>
     if the emotion category label is "ne" (no emotion), there is no emotion intensity label and no emotion indicators
  
* basefile.txt
  contains the sentences corresponding to the annotations in AnnotSet1.txt and AnnotSet2.txt 
  + Format: <id> <sentence>
  
  The numbers in files AnnotSet1.txt, AnnotSet1.txt, and basefile.txt follow the following order:
  
  sentence-numbers        dataset			collected using seed words for
  --------------------------------------------------------------------------------------
 1 - 848 		ec-hp 			happiness
 1 - 884 		ec-sd 			sadness
 1 - 883 		ec-ag 			anger
 1 - 882 		ec-dg 			disgust
 1 - 847 		ec-sp 			surprise
 1 - 861 		ec-fr 			fear
  ---------------------------------------------------------------------------------------
  		
2.1 Annots by A	
	contains annotations perfromed by annotator A
		
2.2 Annots by B
	contains annotations perfromed by annotator B
	
2.3 Annots by C
	contains annotations perfromed by annotator C

2.4 Annots by D
	contains annotations perfromed by annotator D


3. Benchmark

* category_gold_std.txt 
   contains all those sentences for which the annotators agreed on the emotion category
	+ Format: <emotion label> <sentence number> <sentence>

*  intensity_gold_std_hmln.txt 
    contains all those sentences for which the annotators agreed on the emotion intensity
	+ Format: <emotion intensity> <sentence number> <sentence>

---------------------------------------------------------------------------------------------------
Contact Information

Please send any questions about this dataset to 
Stan Szpakowicz (szpak@site.uottawa.ca)
Saima Aman (saimaaman@hotmail.com)

 