# cs445final
In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

The best accuracy for the IIA30 dataset was with contrastive divergence : savemodel/contrasMIXEDsimpledata/netconv25.pth
                  precision    recall  f1-score   support

           other       0.00      0.00      0.00         0
        calendar       1.00      1.00      1.00         5
      ponce_book       1.00      0.50      0.67         2
         red_cup       1.00      1.00      1.00         2
            rack       1.00      1.00      1.00         3
   poster_spices       1.00      1.00      1.00         3
        monitor1       1.00      1.00      1.00         4
          chair2       1.00      1.00      1.00         2
poster_mystrands       1.00      1.00      1.00         5
           cube3       1.00      1.00      1.00         3
         charger       1.00      1.00      1.00         1
        monitor3       1.00      1.00      1.00         6
         bicycle       1.00      1.00      1.00         4
          chair1       1.00      1.00      1.00         4
    extinguisher       1.00      1.00      1.00         3
    gray_battery       1.00      1.00      1.00         4
           cube1       1.00      0.75      0.86         4
           cube2       0.80      1.00      0.89         4
    hartley_book       1.00      1.00      1.00         4
        monitor2       1.00      1.00      1.00         2
      dentifrice       1.00      1.00      1.00         2
           phone       1.00      1.00      1.00         6
          window       1.00      1.00      1.00        10
     poster_cmpi       0.89      1.00      0.94         8
         stapler       1.00      1.00      1.00         4
       orbit_box       1.00      1.00      1.00         7
     red_battery       1.00      1.00      1.00         2
        red_wine       1.00      0.67      0.80         3
          chair3       1.00      1.00      1.00         5
        umbrella       1.00      0.50      0.67         2

     avg / total       0.99      0.96      0.97       114

0.964912280702

