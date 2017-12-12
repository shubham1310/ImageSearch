In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

CAL101 : Contrastive best :  savemodel/CAL101/netconv50.pth (checked till 60)

                   precision    recall  f1-score   support

            other       1.00      1.00      1.00        16
       Faces_easy       1.00      0.50      0.67        16
            pizza       1.00      0.61      0.76        18
          gerenuk       1.00      0.74      0.85        19
            llama       1.00      1.00      1.00        16
           camera       0.68      0.95      0.79        20
            lotus       1.00      0.90      0.95        20
        euphonium       0.46      0.69      0.55        16
              ant       0.90      0.64      0.75        14
          rooster       0.61      0.65      0.63        17
           mayfly       1.00      0.44      0.62        18
            panda       0.00      0.00      0.00        12
          lobster       0.56      0.62      0.59         8
         schooner       0.44      0.94      0.60        16
       chandelier       0.88      0.88      0.88        17
          menorah       1.00      0.65      0.79        23
        saxophone       1.00      0.87      0.93        23
         flamingo       1.00      0.53      0.69        17
   crocodile_head       1.00      0.83      0.91        12
       wheelchair       0.59      0.80      0.68        20
            chair       1.00      0.71      0.83        17
        dragonfly       1.00      0.62      0.76        13
        binocular       0.21      0.75      0.32        16
BACKGROUND_Google       0.53      0.62      0.57        13
          octopus       0.32      0.60      0.42        15
             crab       0.32      0.47      0.38        19
        crocodile       1.00      0.83      0.91        18
         mandolin       1.00      0.57      0.73        14
      dollar_bill       1.00      0.89      0.94        18
      stegosaurus       1.00      0.61      0.76        18
     inline_skate       0.95      1.00      0.98        20
           bonsai       1.00      0.63      0.77        19
            okapi       0.80      1.00      0.89        12
    windsor_chair       1.00      0.29      0.45        17
      cougar_body       1.00      0.25      0.40        12
       helicopter       1.00      1.00      1.00        18
         hedgehog       0.74      0.81      0.77        21
        trilobite       0.94      0.77      0.85        22
        cellphone       0.40      0.57      0.47        14
          dolphin       0.57      0.86      0.69        14
         elephant       1.00      0.94      0.97        16
              emu       1.00      1.00      1.00        22
        accordion       0.94      1.00      0.97        16
             ewer       1.00      0.81      0.90        27
        sunflower       1.00      0.91      0.95        23
             tick       0.46      0.71      0.56        17
            brain       0.71      0.91      0.80        11
             ibis       1.00      0.70      0.82        20
           pagoda       0.50      0.90      0.64        10
    flamingo_head       1.00      0.20      0.33        10
         wild_cat       0.67      0.40      0.50        15
           beaver       1.00      1.00      1.00        23
            Faces       0.94      0.94      0.94        17
      soccer_ball       0.68      0.57      0.62        23
             lamp       1.00      0.93      0.97        15
         yin_yang       1.00      0.71      0.83        14
       gramophone       0.55      0.67      0.60         9
           anchor       0.59      0.94      0.73        17
         kangaroo       0.76      1.00      0.86        22
         garfield       0.61      0.55      0.58        20
           wrench       0.71      0.52      0.60        23
           snoopy       1.00      0.31      0.48        16
           cannon       1.00      0.85      0.92        20
         scissors       0.45      1.00      0.62        14
        airplanes       0.60      0.38      0.46        16
         crayfish       0.75      0.30      0.43        20
      ceiling_fan       0.91      0.53      0.67        19
      water_lilly       0.95      0.91      0.93        23
        dalmatian       0.44      0.47      0.45        15
     brontosaurus       0.50      0.62      0.56         8
              cup       0.50      0.55      0.52        11
        butterfly       1.00      0.89      0.94        18
  electric_guitar       0.94      0.80      0.86        20
      grand_piano       1.00      0.35      0.52        17
            rhino       0.78      1.00      0.88        18
          minaret       1.00      0.89      0.94        18
      cougar_face       0.67      0.67      0.67        15
            ketch       0.55      0.60      0.57        20
        sea_horse       1.00      0.65      0.79        17
        metronome       0.76      0.62      0.68        21
           pigeon       0.64      0.95      0.77        19
       Motorbikes       0.53      0.95      0.68        22
            watch       0.81      0.93      0.87        14
        headphone       1.00      1.00      1.00        14
        stop_sign       0.93      0.78      0.85        18
          stapler       1.00      0.74      0.85        19
       strawberry       1.00      0.69      0.81        16
         revolver       0.92      0.79      0.85        14
          pyramid       1.00      0.92      0.96        13
           laptop       0.75      0.18      0.29        17
           barrel       0.40      0.60      0.48        10
         scorpion       0.67      0.91      0.77        11
           buddha       0.37      0.58      0.45        12
         platypus       1.00      0.81      0.89        26
        hawksbill       0.40      0.95      0.57        20
         Leopards       1.00      1.00      1.00        17
         nautilus       0.85      0.73      0.79        15
             bass       0.61      0.64      0.62        22
         starfish       0.53      0.82      0.64        11
         umbrella       0.81      1.00      0.89        25
      joshua_tree       1.00      0.80      0.89        10

      avg / total       0.81      0.74      0.74      1709

0.739028671738



CAL101 : Dot Product best :  savemodel/CAL101dot/netconv23.pth (checked till 28)

                   precision    recall  f1-score   support

            other       0.75      1.00      0.86        18
       Faces_easy       0.94      0.88      0.91        17
            pizza       0.77      0.50      0.61        20
          gerenuk       0.71      0.55      0.62        22
            llama       1.00      1.00      1.00         9
           camera       0.35      0.55      0.43        11
            lotus       0.94      0.89      0.92        19
        euphonium       0.68      0.79      0.73        19
              ant       1.00      0.82      0.90        11
          rooster       0.66      0.88      0.75        24
           mayfly       1.00      0.75      0.86        16
            panda       0.95      0.69      0.80        26
          lobster       0.69      0.60      0.64        15
         schooner       0.92      0.92      0.92        24
       chandelier       1.00      0.83      0.91        12
          menorah       0.89      0.89      0.89         9
        saxophone       0.80      0.60      0.69        20
         flamingo       0.48      0.77      0.59        13
   crocodile_head       1.00      1.00      1.00        23
       wheelchair       1.00      0.76      0.86        21
            chair       1.00      0.82      0.90        17
        dragonfly       0.89      0.89      0.89        19
        binocular       0.18      0.55      0.27        11
BACKGROUND_Google       0.64      0.47      0.54        15
          octopus       0.18      0.37      0.25        19
             crab       0.29      0.41      0.34        17
        crocodile       1.00      0.67      0.80        21
         mandolin       1.00      1.00      1.00        16
      dollar_bill       1.00      0.67      0.80        18
      stegosaurus       1.00      1.00      1.00        16
     inline_skate       0.86      1.00      0.93        19
           bonsai       0.84      1.00      0.91        16
            okapi       0.62      1.00      0.76         8
    windsor_chair       0.57      0.71      0.63        17
      cougar_body       0.61      0.85      0.71        13
       helicopter       1.00      0.83      0.90        23
         hedgehog       1.00      0.95      0.97        20
        trilobite       0.85      0.94      0.89        18
        cellphone       1.00      1.00      1.00        18
          dolphin       0.83      0.83      0.83        18
         elephant       0.81      1.00      0.90        13
              emu       1.00      1.00      1.00        15
        accordion       1.00      0.83      0.91        18
             ewer       1.00      0.69      0.82        13
        sunflower       1.00      0.80      0.89        15
             tick       0.76      1.00      0.86        22
            brain       0.67      0.84      0.74        19
             ibis       0.94      0.88      0.91        17
           pagoda       1.00      0.92      0.96        12
    flamingo_head       0.96      0.96      0.96        23
         wild_cat       0.64      0.82      0.72        11
           beaver       1.00      0.68      0.81        19
            Faces       1.00      0.95      0.97        19
      soccer_ball       0.79      0.85      0.81        13
             lamp       0.75      0.78      0.77        23
         yin_yang       1.00      1.00      1.00        19
       gramophone       0.89      0.76      0.82        21
           anchor       0.55      0.63      0.59        19
         kangaroo       0.72      0.93      0.81        14
         garfield       0.93      0.61      0.74        23
           wrench       1.00      0.59      0.74        17
           snoopy       1.00      0.62      0.77        16
           cannon       1.00      1.00      1.00        17
         scissors       0.89      1.00      0.94        17
        airplanes       0.30      0.16      0.21        19
         crayfish       0.70      0.73      0.71        22
      ceiling_fan       0.58      0.47      0.52        15
      water_lilly       1.00      1.00      1.00        16
        dalmatian       0.75      0.46      0.57        13
     brontosaurus       0.73      0.94      0.82        17
              cup       0.61      0.85      0.71        13
        butterfly       1.00      0.77      0.87        13
  electric_guitar       1.00      1.00      1.00        18
      grand_piano       0.56      0.45      0.50        11
            rhino       0.93      1.00      0.96        13
          minaret       1.00      0.88      0.94        25
      cougar_face       0.57      0.67      0.62        12
            ketch       0.56      0.75      0.64        24
        sea_horse       1.00      1.00      1.00        13
        metronome       1.00      0.86      0.92        14
           pigeon       1.00      1.00      1.00        13
       Motorbikes       0.83      0.94      0.88        16
            watch       0.84      0.89      0.86        18
        headphone       0.86      1.00      0.92        18
        stop_sign       1.00      0.95      0.97        19
          stapler       0.95      0.90      0.92        20
       strawberry       1.00      0.93      0.97        15
         revolver       1.00      1.00      1.00        18
          pyramid       1.00      1.00      1.00        16
           laptop       1.00      0.44      0.61        16
           barrel       0.91      0.83      0.87        12
         scorpion       1.00      0.85      0.92        13
           buddha       1.00      0.77      0.87        13
         platypus       0.86      0.75      0.80        16
        hawksbill       0.62      0.91      0.74        11
         Leopards       1.00      0.94      0.97        17
         nautilus       0.73      0.70      0.71        23
             bass       0.90      0.70      0.79        27
         starfish       0.88      0.82      0.85        17
         umbrella       0.80      1.00      0.89        16
      joshua_tree       1.00      0.86      0.92        14

      avg / total       0.84      0.81      0.82      1709

0.808660035108










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

Best accuracy for dot loss: savemodel/newdatadotprod/netconv53.pth

|    precision  |  recall | f1-score  | support|
|---------|------|------| ------|
|           other |      0.00   |   0.00  |    0.00   |      0|
|        calendar |      1.00   |   1.00  |    1.00   |      3|
|      ponce_book |      1.00   |   1.00  |    1.00   |      2|
|         red_cup |      1.00   |   1.00  |    1.00   |      7|
|            rack  |     1.00   |   1.00    |  1.00     |    7|
|   poster_spices  |     1.00   |   1.00    |  1.00     |    2|
|        monitor1  |     1.00   |   1.00    |  1.00     |    2|
|          chair2  |     1.00   |   1.00    |  1.00     |    2|
|poster_mystrands  |    1.00    |  1.00   |   1.00    |     6|
|           cube3  |    1.00    |  0.50   |   0.67    |     4|
|         charger  |    1.00    |  1.00   |   1.00    |     4|
|        monitor3  |    1.00    |  1.00   |   1.00    |     4|
|         bicycle  |     1.00   |   1.00  |    1.00    |     3|
|          chair1  |     1.00   |   1.00  |    1.00    |     2|
|    extinguisher  |     1.00   |   1.00  |    1.00    |     8|
|    gray_battery  |     1.00   |   1.00  |    1.00    |     4|
|           cube1  |     0.00   |   0.00  |    0.00    |     0|
|           cube2  |     1.00   |   1.00  |    1.00    |     2|
|    hartley_book  |     1.00   |   1.00  |    1.00    |     4|
|        monitor2  |     1.00   |   1.00  |    1.00    |     3|
|      dentifrice    |   1.00    |  1.00    |  1.00    |     5|
|           phone    |   1.00    |  1.00    |  1.00    |     1|
|          window    |   1.00    |  1.00    |  1.00    |     3|
|     poster_cmpi    |   1.00    |  1.00    |  1.00    |     9|
|         stapler   |    1.00   |   1.00   |   1.00    |     6|
|       orbit_box   |    1.00   |   1.00   |   1.00    |     7|
|     red_battery   |    1.00   |   0.80   |   0.89    |     5|
|        red_wine   |    1.00   |   1.00   |   1.00    |     5|
|          chair3     |  1.00      |1.00      |1.00    |     4|
|     avg / total     |  1.00    |  0.97   |   0.98    |   114|

0.973684210526


Best accuracy for neural loss: savemodel/newdataneural/netconv61.pth

|                  precision|    recall | f1-score  | support  |
|---------|------|------| ------|
|           other    |   0.57   |   1.00   |   0.73   |      4|
|        calendar    |   0.50   |   0.60   |   0.55   |      5|
|      ponce_book    |   1.00   |   1.00   |   1.00   |      2|
|         red_cup    |   1.00   |   1.00   |   1.00   |      4|
|            rack    |   1.00   |   1.00   |   1.00   |      3|
|   poster_spices    |   1.00   |   0.67   |   0.80   |      6|
|        monitor1    |   1.00   |   0.33   |   0.50   |      6|
|          chair2    |   1.00   |   1.00   |   1.00   |      3|
|poster_mystrands    |   0.14   |   0.50   |   0.22   |      2|                                                                
|           cube3    |   0.33   |   0.50   |   0.40   |      2|                                                                       
|         charger    |   0.75   |   1.00   |   0.86   |      6|                                                                     
|        monitor3    |   1.00   |   1.00   |   1.00   |      3|                                                               
|         bicycle    |   0.80   |   1.00   |   0.89   |      4|                                                                 
|          chair1    |   0.86   |   1.00   |   0.92   |      6|                                                                 
|    extinguisher    |   0.00   |   0.00   |   0.00   |      1|                                                                 
|    gray_battery    |   1.00   |   0.33   |   0.50   |      6|                                                                
|           cube1    |   0.33   |   0.33   |   0.33   |      3|                                                               
|           cube2    |   1.00   |   1.00   |   1.00   |      7|                                                               
|    hartley_book    |   1.00   |   0.50   |   0.67   |      2|                                                        
|        monitor2    |   1.00   |   1.00   |   1.00   |      4|                                                                 
|      dentifrice    |   1.00   |   1.00   |   1.00   |      1|                                                               
|           phone    |   1.00   |   1.00   |   1.00   |     11|                                                                    
|          window    |   0.50   |   0.40   |   0.44   |      5|                                                                   
|     poster_cmpi    |   0.00   |   0.00   |   0.00   |      4|                                                              
|         stapler    |   1.00   |   1.00   |   1.00   |      2|                                                                  
|       orbit_box    |   0.00   |   0.00   |   0.00   |      1|                                                                
|     red_battery    |   1.00   |   0.86   |   0.92   |      7|                                                              
|        red_wine    |   1.00   |   0.50   |   0.67   |      2|                                                  
|          chair3    |   0.17   |   0.50   |   0.25   |      2|              
|     avg / total    |   0.80   |   0.75   |   0.74   |    114|

0.745614035088