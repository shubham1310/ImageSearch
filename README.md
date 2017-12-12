In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

CAL101 : Contrastive best :  savemodel/CAL101/netconv99.pth (checked from 108 till 97)

Using 3 number of neighbours

                   precision    recall  f1-score   support

            other       1.00      0.94      0.97        17
       Faces_easy       1.00      0.68      0.81        19
            pizza       1.00      0.36      0.53        14
          gerenuk       1.00      0.72      0.84        18
            llama       1.00      1.00      1.00        22
           camera       0.67      0.95      0.78        19
            lotus       1.00      1.00      1.00        15
        euphonium       0.52      0.88      0.65        16
              ant       0.93      0.74      0.82        19
          rooster       0.72      0.76      0.74        17
           mayfly       1.00      0.37      0.54        19
            panda       0.00      0.00      0.00        16
          lobster       0.82      0.82      0.82        17
         schooner       0.49      0.86      0.62        22
       chandelier       0.90      0.82      0.86        22
          menorah       1.00      0.40      0.57        20
        saxophone       1.00      0.93      0.96        14
         flamingo       1.00      0.56      0.71        18
   crocodile_head       1.00      0.88      0.93        16
       wheelchair       0.63      0.86      0.73        14
            chair       1.00      0.44      0.61        16
        dragonfly       1.00      0.81      0.90        16
        binocular       0.25      0.74      0.37        19
BACKGROUND_Google       0.80      0.70      0.74        23
          octopus       0.36      0.62      0.45        16
             crab       0.22      0.43      0.29        14
        crocodile       1.00      0.65      0.79        20
         mandolin       1.00      1.00      1.00        12
      dollar_bill       1.00      0.80      0.89        10
      stegosaurus       1.00      0.63      0.77        19
     inline_skate       0.76      1.00      0.86        19
           bonsai       1.00      0.47      0.64        15
            okapi       0.92      1.00      0.96        12
    windsor_chair       1.00      0.69      0.81        16
      cougar_body       1.00      0.53      0.70        15
       helicopter       1.00      0.81      0.90        16
         hedgehog       0.87      0.76      0.81        17
        trilobite       0.93      0.70      0.80        20
        cellphone       0.39      0.78      0.52         9
          dolphin       0.78      0.93      0.85        15
         elephant       0.76      0.93      0.84        14
              emu       1.00      1.00      1.00        16
        accordion       0.95      0.95      0.95        19
             ewer       1.00      0.74      0.85        19
        sunflower       1.00      0.95      0.98        22
             tick       0.54      0.94      0.68        16
            brain       0.88      0.88      0.88        17
             ibis       0.82      0.82      0.82        11
           pagoda       0.78      0.82      0.80        17
    flamingo_head       1.00      0.80      0.89        15
         wild_cat       0.92      0.48      0.63        23
           beaver       0.93      1.00      0.97        14
            Faces       0.90      0.90      0.90        20
      soccer_ball       0.86      0.43      0.57        14
             lamp       1.00      0.89      0.94        18
         yin_yang       1.00      0.76      0.87        17
       gramophone       0.63      0.55      0.59        22
           anchor       0.70      1.00      0.83        19
         kangaroo       0.68      1.00      0.81        17
         garfield       0.44      0.44      0.44         9
           wrench       0.73      0.65      0.69        17
           snoopy       1.00      0.40      0.57        15
           cannon       1.00      0.78      0.88        18
         scissors       0.35      1.00      0.52        16
        airplanes       0.69      0.60      0.64        15
         crayfish       0.69      0.56      0.62        16
      ceiling_fan       0.90      0.53      0.67        17
      water_lilly       0.86      0.95      0.90        20
        dalmatian       0.38      0.71      0.50        14
     brontosaurus       0.78      0.67      0.72        21
              cup       0.51      0.75      0.61        24
        butterfly       1.00      0.86      0.92        14
  electric_guitar       0.95      0.91      0.93        22
      grand_piano       1.00      0.47      0.64        17
            rhino       0.89      0.94      0.92        18
          minaret       1.00      0.85      0.92        13
      cougar_face       0.77      0.62      0.69        16
            ketch       0.48      0.85      0.61        13
        sea_horse       1.00      0.95      0.98        21
        metronome       0.92      0.80      0.86        15
           pigeon       0.56      1.00      0.72        18
       Motorbikes       0.54      1.00      0.70        14
            watch       0.77      0.83      0.80        12
        headphone       0.89      1.00      0.94        16
        stop_sign       0.93      0.70      0.80        20
          stapler       1.00      0.46      0.63        13
       strawberry       1.00      0.87      0.93        15
         revolver       0.83      0.83      0.83        12
          pyramid       1.00      0.85      0.92        20
           laptop       0.50      0.27      0.35        15
           barrel       0.65      0.62      0.63        21
         scorpion       0.76      0.87      0.81        15
           buddha       0.44      0.47      0.46        17
         platypus       0.91      0.67      0.77        15
        hawksbill       0.48      0.83      0.61        18
         Leopards       1.00      0.79      0.88        14
         nautilus       0.81      0.72      0.76        18
             bass       0.85      0.68      0.76        25
         starfish       0.68      0.65      0.67        20
         umbrella       0.90      1.00      0.95        19
      joshua_tree       1.00      0.78      0.88        18

      avg / total       0.82      0.75      0.76      1709

0.753657109421

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



CAL101 : Neural best :  avemodel/CAL101neural/netconv20.pth (checked till 33)



                   precision    recall  f1-score   support

            other       0.80      1.00      0.89        16
       Faces_easy       0.18      0.12      0.14        17
            pizza       0.10      0.22      0.14         9
          gerenuk       0.81      0.52      0.63        25
            llama       0.50      0.67      0.57        18
           camera       0.69      0.46      0.55        24
            lotus       0.38      0.59      0.47        17
        euphonium       0.00      0.00      0.00        14
              ant       0.36      0.50      0.42        16
          rooster       0.11      0.17      0.13        18
           mayfly       0.22      0.14      0.17        14
            panda       0.27      0.50      0.35        12
          lobster       0.60      0.38      0.46        16
         schooner       0.14      0.17      0.16        23
       chandelier       0.60      0.14      0.23        21
          menorah       0.25      0.06      0.10        17
        saxophone       0.25      0.27      0.26        11
         flamingo       0.18      0.24      0.21        17
   crocodile_head       1.00      0.07      0.13        14
       wheelchair       0.04      0.05      0.04        21
            chair       0.23      0.24      0.23        21
        dragonfly       0.07      0.07      0.07        15
        binocular       0.12      0.08      0.10        12
BACKGROUND_Google       0.28      0.28      0.28        18
          octopus       0.06      0.19      0.09        16
             crab       0.00      0.00      0.00        14
        crocodile       0.47      0.39      0.42        18
         mandolin       0.12      0.45      0.19        11
      dollar_bill       0.27      0.38      0.32        16
      stegosaurus       0.53      0.70      0.60        23
     inline_skate       0.07      0.14      0.09        14
           bonsai       0.11      0.33      0.16         9
            okapi       0.37      0.91      0.53        23
    windsor_chair       0.38      0.19      0.25        16
      cougar_body       0.00      0.00      0.00        13
       helicopter       0.20      0.14      0.17        14
         hedgehog       0.35      0.65      0.46        17
        trilobite       0.00      0.00      0.00        19
        cellphone       0.12      0.19      0.15        16
          dolphin       0.31      0.18      0.23        22
         elephant       0.00      0.00      0.00        12
              emu       0.00      0.00      0.00        15
        accordion       0.14      0.11      0.12        19
             ewer       0.75      0.45      0.56        20
        sunflower       0.50      0.45      0.47        20
             tick       0.15      0.33      0.20        18
            brain       0.75      0.20      0.32        15
             ibis       0.00      0.00      0.00        13
           pagoda       0.50      0.64      0.56        25
    flamingo_head       0.00      0.00      0.00        11
         wild_cat       0.00      0.00      0.00        16
           beaver       1.00      0.31      0.48        16
            Faces       0.40      0.25      0.31        16
      soccer_ball       0.23      0.19      0.21        16
             lamp       0.35      0.35      0.35        20
         yin_yang       0.00      0.00      0.00        17
       gramophone       0.00      0.00      0.00        17
           anchor       0.00      0.00      0.00        17
         kangaroo       0.39      0.50      0.44        22
         garfield       0.50      0.50      0.50        10
           wrench       0.26      0.50      0.34        14
           snoopy       0.20      0.07      0.10        15
           cannon       0.20      0.20      0.20        10
         scissors       0.42      0.44      0.43        18
        airplanes       0.00      0.00      0.00        15
         crayfish       0.00      0.00      0.00        22
      ceiling_fan       0.20      0.18      0.19        11
      water_lilly       1.00      0.11      0.20        18
        dalmatian       0.00      0.00      0.00        18
     brontosaurus       1.00      0.55      0.71        11
              cup       0.05      0.05      0.05        19
        butterfly       0.25      0.19      0.21        16
  electric_guitar       0.57      0.25      0.35        16
      grand_piano       0.00      0.00      0.00        20
            rhino       0.11      0.15      0.13        13
          minaret       0.75      0.17      0.27        18
      cougar_face       0.12      0.19      0.15        16
            ketch       0.00      0.00      0.00        20
        sea_horse       0.55      0.52      0.54        21
        metronome       0.53      0.40      0.46        20
           pigeon       0.74      0.87      0.80        23
       Motorbikes       1.00      0.56      0.71         9
            watch       0.00      0.00      0.00        16
        headphone       0.18      0.38      0.24         8
        stop_sign       0.13      0.08      0.10        25
          stapler       0.00      0.00      0.00        15
       strawberry       0.60      0.30      0.40        20
         revolver       0.27      0.24      0.25        17
          pyramid       0.92      0.42      0.58        26
           laptop       0.00      0.00      0.00        11
           barrel       0.04      0.04      0.04        25
         scorpion       0.00      0.00      0.00        14
           buddha       0.12      0.16      0.14        19
         platypus       0.09      0.24      0.13        17
        hawksbill       0.46      0.52      0.49        23
         Leopards       0.44      0.26      0.33        27
         nautilus       0.07      0.08      0.07        13
             bass       0.20      0.24      0.22        17
         starfish       0.07      0.06      0.06        17
         umbrella       0.25      0.31      0.28        16
      joshua_tree       0.00      0.00      0.00        18

      avg / total       0.30      0.26      0.25      1709

0.256290228204




CAL256 : Contrastive best :  savemodel/CAL256/netconv104.pth (checked from 107 to 104)


Using 9 number of neighbours
Nearest neighbours Classifier trained
Prediction done for 0/186
Prediction done for 50/186
Prediction done for 100/186
Prediction done for 150/186
                               precision    recall  f1-score   support

                        other       1.00      0.64      0.78        22
              094.guitar-pick       0.78      0.24      0.37        29
      063.electric-guitar-101       0.31      0.58      0.40        19
           156.paper-shredder       0.17      0.25      0.20        24
                  096.hammock       0.71      0.26      0.38        19
             235.umbrella-101       0.39      0.39      0.39        18
              246.wine-bottle       0.40      0.07      0.12        27
             174.rotary-phone       0.22      0.69      0.33        16
              057.dolphin-101       0.40      0.15      0.22        26
            226.traffic-light       0.24      0.33      0.28        24
                     117.ipod       0.44      0.40      0.42        30
                    125.knife       0.80      0.25      0.38        16
             062.eiffel-tower       0.67      0.22      0.33        18
           202.steering-wheel       1.00      0.72      0.84        29
               053.desk-globe       0.47      0.71      0.57        24
                   214.teepee       0.06      0.10      0.08        20
                 144.minotaur       0.30      0.62      0.41        21
             217.tennis-court       0.43      0.57      0.49        23
                  084.giraffe       0.11      0.09      0.10        23
                    028.camel       0.43      0.38      0.41        26
             054.diamond-ring       0.44      0.50      0.47        16
                  079.frisbee       0.80      0.80      0.80        25
                240.watch-101       0.37      0.61      0.46        23
           112.human-skeleton       0.95      0.68      0.79        28
               072.fire-truck       0.76      0.86      0.81        22
                095.hamburger       0.83      0.71      0.77        21
       086.golden-gate-bridge       0.38      0.35      0.36        23
                024.butterfly       0.52      0.48      0.50        29
                   212.teapot       0.88      0.35      0.50        20
              228.triceratops       0.55      0.22      0.32        27
                111.house-fly       0.18      0.17      0.17        18
                134.llama-101       0.70      0.29      0.41        24
             121.kangaroo-101       0.52      0.70      0.60        20
                  220.toaster       0.50      0.33      0.40        27
                  210.syringe       0.36      0.23      0.28        22
                   231.tripod       0.17      0.12      0.14        25
             163.playing-card       0.06      0.11      0.07        19
                 013.birdbath       0.21      0.35      0.26        26
                  236.unicorn       0.79      0.44      0.56        25
                   082.galaxy       0.30      0.35      0.33        20
                     207.swan       0.71      0.75      0.73        16
                023.bulldozer       0.81      0.50      0.62        26
                     256.toad       0.50      0.41      0.45        17
                011.billiards       0.36      0.46      0.41        26
              132.light-house       0.29      0.52      0.37        21
                  183.sextant       0.54      0.61      0.57        23
                  170.rainbow       0.89      0.89      0.89        19
               178.school-bus       0.32      0.24      0.27        29
                 229.tricycle       0.34      0.48      0.40        21
               015.bonsai-101       0.33      0.48      0.39        23
                078.fried-egg       0.18      0.22      0.20        23
                 245.windmill       0.76      0.42      0.54        31
                 083.gas-pump       0.14      0.21      0.17        24
                     026.cake       1.00      0.38      0.55        21
           175.roulette-wheel       0.22      0.36      0.27        28
            160.pez-dispenser       0.26      0.33      0.29        21
              069.fighter-jet       0.27      0.29      0.28        24
           005.baseball-glove       0.13      0.16      0.14        19
             064.elephant-101       0.70      0.55      0.62        29
                040.cockroach       0.85      0.63      0.72        27
               213.teddy-bear       0.45      0.23      0.30        22
         208.swiss-army-knife       0.89      0.84      0.86        19
                     098.harp       0.19      0.16      0.17        19
                     009.bear       1.00      0.62      0.77        24
                164.porcupine       0.83      0.45      0.59        22
                 103.hibiscus       0.12      0.14      0.13        21
                  148.mussels       0.50      0.50      0.50        22
                 114.ibis-101       0.67      0.42      0.51        24
                    038.chimp       0.90      0.45      0.60        20
               022.buddha-101       1.00      0.72      0.84        25
                    250.zebra       0.38      0.36      0.37        22
            002.american-flag       0.40      0.32      0.36        25
                142.microwave       0.67      0.60      0.63        20
            146.mountain-bike       0.91      0.43      0.59        23
                  223.top-hat       0.32      0.42      0.36        24
                155.paperclip       0.07      0.05      0.06        21
               074.flashlight       0.68      0.50      0.58        26
                   176.saddle       1.00      0.32      0.48        25
                  032.cartman       0.14      0.18      0.15        17
                   042.coffin       1.00      0.62      0.76        26
             172.revolver-101       0.77      0.80      0.78        25
              077.french-horn       0.10      0.08      0.09        24
             162.picnic-table       0.11      0.25      0.15         8
                    089.goose       0.86      0.67      0.75        36
                  234.tweezer       0.89      0.42      0.57        19
                  151.ostrich       0.65      0.52      0.58        21
                 003.backpack       0.30      0.53      0.38        17
                227.treadmill       0.07      0.15      0.10        20
                  135.mailbox       0.47      0.32      0.38        22
                    194.socks       0.22      0.33      0.26        24
           166.praying-mantis       0.59      0.50      0.54        32
              099.harpsichord       0.48      0.40      0.43        25
                 136.mandolin       0.15      0.29      0.20        17
                    173.rifle       0.86      0.72      0.78        25
                123.ketch-101       0.28      0.57      0.38        30
                     085.goat       0.78      0.33      0.47        21
             201.starfish-101       1.00      0.70      0.83        27
            230.trilobite-101       0.28      0.41      0.33        22
             019.boxing-glove       0.23      0.13      0.17        38
               211.tambourine       0.42      0.31      0.36        35
           106.horseshoe-crab       0.38      0.38      0.38        24
          006.basketball-hoop       0.17      0.35      0.23        23
                    048.conch       0.95      0.80      0.87        25
            204.sunflower-101       0.38      0.52      0.44        23
                  090.gorilla       0.87      0.54      0.67        24
               225.tower-pisa       0.12      0.50      0.19        20
                  257.clutter       0.15      0.28      0.20        18
                   025.cactus       0.17      0.19      0.18        27
                 147.mushroom       0.80      0.28      0.41        29
               041.coffee-mug       0.94      0.70      0.80        23
               067.eyeglasses       0.95      0.90      0.92        20
                     137.mars       1.00      0.19      0.31        27
             255.tennis-shoes       0.27      0.50      0.35        16
             004.baseball-bat       0.62      0.65      0.63        20
        045.computer-keyboard       0.24      0.23      0.24        35
                    105.horse       0.35      0.42      0.38        19
                139.megaphone       0.52      0.63      0.57        19
         046.computer-monitor       0.44      0.27      0.33        26
                 016.boom-box       0.32      0.38      0.35        29
                    190.snake       0.67      0.47      0.55        34
                196.spaghetti       0.60      0.33      0.43        18
          239.washing-machine       0.10      0.14      0.12        14
                 087.goldfish       0.81      0.86      0.83        29
              037.chess-board       0.69      0.69      0.69        29
                 248.yarmulke       0.32      0.32      0.32        19
                     043.coin       0.55      0.63      0.59        19
               021.breadmaker       0.22      0.15      0.18        26
              093.grasshopper       0.07      0.26      0.11        23
               185.skateboard       0.35      0.23      0.28        26
              180.screwdriver       0.44      0.25      0.32        16
                       033.cd       0.60      0.11      0.18        28
                     068.fern       0.65      0.48      0.55        23
                  109.hot-tub       0.43      0.55      0.48        22
                   092.grapes       0.70      0.29      0.41        24
           102.helicopter-101       0.21      0.24      0.22        17
                    209.sword       0.27      0.20      0.23        15
               141.microscope       0.50      0.22      0.30        23
                 205.superman       0.33      0.33      0.33        27
                247.xylophone       0.92      1.00      0.96        22
             129.leopards-101       0.29      0.33      0.31        27
                 138.mattress       0.17      0.18      0.18        22
                    249.yo-yo       1.00      0.42      0.59        24
                      237.vcr       0.15      0.33      0.21        15
                    206.sushi       0.55      0.38      0.44        16
             179.scorpion-101       0.15      0.13      0.14        23
                061.dumb-bell       0.44      0.19      0.27        21
               197.speed-boat       0.29      0.64      0.40        22
              101.head-phones       0.33      0.36      0.35        25
                   116.iguana       0.72      0.48      0.58        27
            050.covered-wagon       0.88      0.45      0.60        31
              184.sheet-music       0.28      0.43      0.34        23
                   029.cannon       0.80      0.57      0.67        21
            100.hawksbill-101       0.64      0.35      0.45        26
                   221.tomato       0.22      0.67      0.33        15
                  232.t-shirt       0.65      0.57      0.60        23
                 157.pci-card       0.37      0.29      0.33        24
                 052.crab-101       0.56      0.53      0.54        19
                073.fireworks       0.48      0.39      0.43        28
                154.palm-tree       0.86      0.94      0.90        34
           145.motorbikes-101       0.21      0.29      0.24        24
                  150.octopus       0.75      0.14      0.23        22
                049.cormorant       0.27      0.18      0.22        22
             119.jesus-christ       0.33      0.36      0.35        25
              018.bowling-pin       0.36      0.39      0.38        33
                  167.pyramid       0.41      0.55      0.47        20
        070.fire-extinguisher       0.63      0.50      0.56        24
              216.tennis-ball       0.81      0.57      0.67        23
               127.laptop-101       0.44      0.27      0.33        15
               012.binoculars       0.70      0.70      0.70        20
                110.hourglass       0.08      0.15      0.10        27
                   159.people       0.18      0.07      0.11        27
          169.radio-telescope       0.80      0.16      0.27        25
                034.centipede       0.46      0.33      0.39        18
                222.tombstone       0.73      0.52      0.61        21
                  143.minaret       0.10      0.21      0.14        19
                 058.doorknob       0.35      0.43      0.38        21
                088.golf-ball       1.00      0.48      0.65        25
             224.touring-bike       0.38      0.19      0.25        16
                020.brain-101       0.75      0.43      0.55        21
               081.frying-pan       0.59      0.25      0.35        40
               219.theodolite       0.00      0.00      0.00        23
                     055.dice       0.71      0.46      0.56        26
            215.telephone-box       0.57      0.29      0.38        28
                    128.lathe       0.12      0.17      0.14        29
                   198.spider       0.29      0.45      0.35        20
             171.refrigerator       0.15      0.50      0.23        12
                131.lightbulb       0.47      0.61      0.53        23
                 031.car-tire       0.80      0.55      0.65        22
                133.lightning       0.34      0.57      0.43        21
               242.watermelon       0.60      0.68      0.64        22
                    044.comet       0.54      0.27      0.36        26
                 203.stirrups       0.69      0.56      0.62        16
                     001.ak47       0.24      0.29      0.26        28
                  149.necktie       0.11      0.27      0.15        15
           047.computer-mouse       1.00      0.45      0.62        22
            130.license-plate       0.62      0.65      0.63        20
                097.harmonica       0.65      0.44      0.52        25
          238.video-projector       0.67      0.37      0.48        27
                    014.blimp       0.05      0.05      0.05        21
                  108.hot-dog       0.47      0.64      0.54        25
                     118.iris       0.41      0.42      0.42        26
               027.calculator       0.53      0.57      0.55        14
                241.waterfall       0.56      0.19      0.28        27
               039.chopsticks       0.32      0.29      0.30        24
                     060.duck       0.07      0.12      0.09        16
                    030.canoe       0.09      0.16      0.12        25
           059.drinking-straw       0.50      0.15      0.23        27
               035.cereal-box       0.70      0.28      0.40        25
                 066.ewer-101       0.30      0.37      0.33        19
             017.bowling-ball       0.04      0.13      0.06        23
                   126.ladder       0.73      0.35      0.47        23
          107.hot-air-balloon       1.00      0.48      0.65        27
          091.grand-piano-101       0.46      0.73      0.56        15
               051.cowboy-hat       0.18      0.26      0.21        19
                120.joy-stick       0.21      0.41      0.28        17
                  191.sneaker       0.36      0.23      0.28        22
                      056.dog       0.61      0.74      0.67        23
            104.homer-simpson       0.29      0.24      0.26        25
                    122.kayak       0.89      1.00      0.94        25
           253.faces-easy-101       0.82      0.58      0.68        24
           036.chandelier-101       0.58      0.23      0.33        30
               153.palm-pilot       0.17      0.39      0.24        23
                      007.bat       0.16      0.24      0.19        25
              193.soccer-ball       1.00      0.50      0.67        16
182.self-propelled-lawn-mower       1.00      0.74      0.85        35
                    186.skunk       0.88      0.26      0.40        27
                      065.elk       0.16      0.11      0.13        27
             071.fire-hydrant       0.20      0.41      0.27        22
                    199.spoon       1.00      0.57      0.72        23
                   177.saturn       0.58      0.37      0.45        30
          076.football-helmet       0.12      0.15      0.13        20
               188.smokestack       0.54      0.46      0.50        28
                 010.beer-mug       0.65      0.48      0.55        23
                      152.owl       0.02      0.03      0.03        31
                 195.soda-can       0.22      0.20      0.21        30
                  008.bathtub       0.62      0.70      0.65        23
                   181.segway       0.38      0.17      0.23        18
               192.snowmobile       0.93      0.58      0.72        24
              113.hummingbird       1.00      0.52      0.69        21
              161.photocopier       0.69      0.38      0.49        29
             124.killer-whale       0.11      0.21      0.15        14
                     080.frog       0.59      0.74      0.65        23
            251.airplanes-101       0.61      0.56      0.58        36
                     165.pram       0.16      0.25      0.20        20
              244.wheelbarrow       0.33      0.47      0.38        30
              075.floppy-disk       0.17      0.20      0.18        20
              233.tuning-fork       0.43      0.47      0.45        19
                    189.snail       0.83      0.19      0.30        27
               187.skyscraper       0.56      0.38      0.45        26
                254.greyhound       0.15      0.33      0.21        18
             243.welding-mask       0.16      0.19      0.17        26
           115.ice-cream-cone       0.27      0.38      0.32        24
                  158.penguin       0.43      0.56      0.49        18
            200.stained-glass       0.44      0.54      0.48        13
              140.menorah-101       0.48      0.55      0.52        29
            218.tennis-racket       0.70      0.76      0.73        25

                  avg / total       0.50      0.41      0.43      5942

0.407943453383




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