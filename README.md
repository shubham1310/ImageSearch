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

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

