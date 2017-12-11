In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

The best accuracy for the IIA30 dataset (+ other samples) was with contrastive divergence : savemodel/contrasMIXEDsimpledata/netconv22.pth

|precision |recall|f1 score| support|
|---------|------|------| ------|
|           other   |   0.00   |   0.00    |  0.00     |    0|
|        calendar   |   1.00   |   1.00    |  1.00     |    3|
|      ponce_book   |   1.00   | 1.00      1|.00         |2|
|         red_cup   |   1.00   | 1.00      1|.00         |4|
|            rack   |   1.00   |   1.00   |   1.00    |     5|
|   poster_spices   |   1.00   | 1.00      |1.00        | 7|
|        monitor1   |    1.00  |   1.00     | 1.00      |   6|
|          chair2   |    1.00   |   1.00     | 1.00       |  2|
|poster_mystrands   |    1.00   |   1.00     | 1.00       |  2|
|           cube3   |    0.83   |   1.00     | 0.91       |  5|
|         charger   |    1.00   |   1.00     | 1.00       |  3|
|        monitor3   |    1.00   |   1.00     | 1.00      |   8|
|         bicycle   |    1.00   |   1.00     | 1.00      |   3|
|          chair1   |    1.00   |   1.00     | 1.00      |   4|
|    extinguisher   |    1.00   |   1.00     | 1.00      |   5|
|    gray_battery   |    0.86   |   1.00      |0.92     |    6|
|           cube1   |    1.00   |   0.50      |0.67     |    2|
|           cube2   |    1.00   |   1.00      |1.00     |    5|
|    hartley_book   |    1.00   |   1.00      |1.00     |    5|
|        monitor2   |    1.00   |   0.75      |0.86    |     4|
|      dentifrice   |    1.00   |   1.00      |1.00    |     1|
|           phone   |    1.00   |   1.00      |1.00    |     1|
|          window   |    1.00   |   1.00      |1.00    |     4|
|     poster_cmpi   |    1.00   |   1.00    |  1.00   |      5|
|         stapler   |    1.00   |   1.00    |  1.00   |      4|
|       orbit_box    |   1.00    |  1.00     | 1.00    |     5|
|    red_battery    |   1.00    |  0.83     | 0.91    |     6|
|       red_wine    |   1.00    |  1.00     | 1.00   |      2|
|         chair3    |   1.00    |  1.00     | 1.00   |      4|
|       umbrella    |   1.00    |  1.00     | 1.00   |      1|
|     avg / total    |   0.99   |   0.97    |  0.98  |     114|

0.973684210526

Best accuracy for neural loss: savemodel/newdataneural/netconv61.pth

                  precision    recall  f1-score   support                                                                                                                     
                                                                                                                                                                              
           other       0.57      1.00      0.73       4
        calendar       0.50      0.60      0.55         5
      ponce_book       1.00      1.00      1.00         2
         red_cup       1.00      1.00      1.00         4
            rack       1.00      1.00      1.00         3
   poster_spices       1.00      0.67      0.80         6
        monitor1       1.00      0.33      0.50         6
          chair2       1.00      1.00      1.00         3
poster_mystrands       0.14      0.50      0.22         2                                                                                                                     
           cube3       0.33      0.50      0.40         2                                                                                                                     
         charger       0.75      1.00      0.86         6                                                                                                                     
        monitor3       1.00      1.00      1.00         3                                                                                                                     
         bicycle       0.80      1.00      0.89         4                                                                                                                     
          chair1       0.86      1.00      0.92         6                                                                                                                     
    extinguisher       0.00      0.00      0.00         1                                                                                                                     
    gray_battery       1.00      0.33      0.50         6                                                                                                                     
           cube1       0.33      0.33      0.33         3                                                                                                                     
           cube2       1.00      1.00      1.00         7                                                                                                                     
    hartley_book       1.00      0.50      0.67         2                                                                                                                     
        monitor2       1.00      1.00      1.00         4                                                                                                                     
      dentifrice       1.00      1.00      1.00         1                                                                                                                     
           phone       1.00      1.00      1.00        11                                                                                                                     
          window       0.50      0.40      0.44         5                                                                                                                     
     poster_cmpi       0.00      0.00      0.00         4                                                                                                                     
         stapler       1.00      1.00      1.00         2                                                                                                                     
       orbit_box       0.00      0.00      0.00         1                                                                                                                     
     red_battery       1.00      0.86      0.92         7                                                                                                                     
        red_wine       1.00      0.50      0.67         2                                                                                                      
          chair3       0.17      0.50      0.25         2                                                                                                                                               
     avg / total       0.80      0.75      0.74       114

0.745614035088