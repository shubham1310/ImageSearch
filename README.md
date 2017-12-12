In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

CAL101 : Contrastive best :  savemodel/CAL101/netconv99.pth
Using 3 number of neighbours

                   precision    recall  f1-score   support
      avg / total       0.82      0.75      0.76      1709

Accuracy = 0.753657109421

CAL101 : Dot Product best :  savemodel/CAL101dot/netconv68.pth
Using 6 number of neighbours

                   precision    recall  f1-score   support
      avg / total       0.88      0.83      0.84      1709

Accuracy = 0.832650672908



CAL101 : Neural best :  savemodel/CAL101neural/netconv20.pth 

                   precision    recall  f1-score   support
      avg / total       0.30      0.26      0.25      1709

Accuracy = 0.256290228204


CAL256 : Contrastive best :  savemodel/CAL256/netconv104.pth
Using 9 number of neighbours

                               precision    recall  f1-score   support
                  avg / total       0.50      0.41      0.43      5942

Accuracy = 0.407943453383


CAL256 : dot product loss best :  savemodel/CAL256dot/netconv63.pth
Using 9 number of neighbours

                               precision    recall  f1-score   support
                  avg / total       0.53      0.50      0.50      5942

Accuracy = 0.503702457085



CAL256 : neural loss best : savemodel/CAL256neural/netconv65.pth
Using 3 number of neighbours

                               precision    recall  f1-score   support
                  avg / total       0.11      0.10      0.09      5942

Accuracy = 0.100134634803


The best accuracy for the IIA30 dataset (+ other samples) was with contrastive divergence : savemodel/contrasMIXEDsimpledata/netconv22.pth

|precision |recall|f1 score| support|
|---|---|---|---|
|  avg / total    |   0.99   |   0.97    |  0.98  |     114|

Accuracy = 0.973684210526

Best accuracy for dot loss: savemodel/newdatadotprod/netconv53.pth

|    precision  |  recall | f1-score  | support|
|---|---|---|---|
|     avg / total     |  1.00    |  0.97   |   0.98    |   114|

Accuracy = 0.973684210526


Best accuracy for neural loss: savemodel/newdataneural/netconv61.pth

|                  precision|    recall | f1-score  | support  |   
|---|---|---|---|    
|     avg / total    |   0.80   |   0.75   |   0.74   |    114|

Accuracy = 0.745614035088