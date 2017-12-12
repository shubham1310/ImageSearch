In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

CAL101 : Contrastive best :  savemodel/CAL101/netconv104.pth (checked from 108 till 100)

Using 4 number of neighbours
                   precision    recall  f1-score   support

            other       1.00      1.00      1.00        12
       Faces_easy       1.00      0.68      0.81        19
            pizza       1.00      0.78      0.88         9
          gerenuk       1.00      0.88      0.93        16
            llama       1.00      1.00      1.00        15
           camera       0.67      1.00      0.80        18
            lotus       1.00      0.85      0.92        20
        euphonium       0.59      0.93      0.72        14
              ant       0.88      0.78      0.82         9
          rooster       0.69      0.65      0.67        17
           mayfly       1.00      0.69      0.81        16
            panda       0.00      0.00      0.00        16
          lobster       0.73      0.65      0.69        17
         schooner       0.44      1.00      0.61        15
       chandelier       0.88      0.71      0.79        21
          menorah       1.00      0.47      0.64        15
        saxophone       1.00      0.90      0.95        21
         flamingo       1.00      0.50      0.67        16
   crocodile_head       1.00      1.00      1.00        15
       wheelchair       0.65      0.79      0.71        14
            chair       1.00      0.67      0.80        12
        dragonfly       1.00      0.95      0.97        20
        binocular       0.22      0.82      0.35        17
BACKGROUND_Google       0.73      0.62      0.67        26
          octopus       0.26      0.45      0.33        22
             crab       0.18      0.40      0.25        10
        crocodile       1.00      0.82      0.90        17
         mandolin       1.00      0.77      0.87        22
      dollar_bill       1.00      0.88      0.94        17
      stegosaurus       1.00      0.88      0.94        17
     inline_skate       0.80      1.00      0.89        24
           bonsai       1.00      0.59      0.74        22
            okapi       0.94      1.00      0.97        15
    windsor_chair       1.00      0.33      0.50        18
      cougar_body       1.00      0.53      0.69        17
       helicopter       1.00      1.00      1.00        13
         hedgehog       0.88      0.88      0.88        26
        trilobite       1.00      0.52      0.69        21
        cellphone       0.92      0.76      0.83        29
          dolphin       0.57      0.75      0.65        16
         elephant       1.00      0.85      0.92        20
              emu       1.00      1.00      1.00        18
        accordion       1.00      0.88      0.94        17
             ewer       1.00      0.83      0.91        12
        sunflower       1.00      1.00      1.00        19
             tick       0.59      0.77      0.67        13
            brain       0.76      0.94      0.84        17
             ibis       0.93      0.82      0.87        17
           pagoda       0.79      0.95      0.86        20
    flamingo_head       1.00      0.37      0.54        19
         wild_cat       0.82      0.47      0.60        19
           beaver       1.00      1.00      1.00        15
            Faces       0.87      0.93      0.90        14
      soccer_ball       1.00      0.47      0.64        17
             lamp       1.00      0.88      0.94        17
         yin_yang       1.00      0.68      0.81        19
       gramophone       0.33      0.27      0.30        11
           anchor       0.59      0.95      0.73        20
         kangaroo       0.79      1.00      0.88        15
         garfield       0.71      0.57      0.63        21
           wrench       0.78      0.39      0.52        18
           snoopy       0.91      0.62      0.74        16
           cannon       1.00      1.00      1.00        13
         scissors       0.38      1.00      0.55        15
        airplanes       0.30      0.20      0.24        15
         crayfish       0.75      0.47      0.58        19
      ceiling_fan       1.00      0.65      0.79        20
      water_lilly       0.82      0.93      0.87        15
        dalmatian       0.36      0.56      0.43         9
     brontosaurus       0.94      0.74      0.83        23
              cup       0.58      0.93      0.72        15
        butterfly       1.00      0.85      0.92        20
  electric_guitar       0.95      0.95      0.95        21
      grand_piano       1.00      0.29      0.45        17
            rhino       0.56      1.00      0.71        10
          minaret       1.00      0.94      0.97        18
      cougar_face       0.48      0.73      0.58        15
            ketch       0.46      0.75      0.57        16
        sea_horse       1.00      1.00      1.00         9
        metronome       0.79      0.79      0.79        14
           pigeon       0.71      1.00      0.83        20
       Motorbikes       0.54      1.00      0.70        14
            watch       0.94      0.88      0.91        17
        headphone       0.84      1.00      0.91        16
        stop_sign       0.75      0.69      0.72        13
          stapler       1.00      0.73      0.84        11
       strawberry       1.00      0.88      0.93        16
         revolver       1.00      0.42      0.59        19
          pyramid       1.00      0.76      0.86        25
           laptop       0.73      0.38      0.50        21
           barrel       0.40      0.57      0.47        14
         scorpion       0.64      0.53      0.58        17
           buddha       0.47      0.47      0.47        17
         platypus       0.93      0.78      0.85        18
        hawksbill       0.33      1.00      0.50        12
         Leopards       1.00      0.86      0.92        14
         nautilus       1.00      0.50      0.67        14
             bass       0.55      0.67      0.60        18
         starfish       0.69      0.78      0.73        23
         umbrella       0.88      1.00      0.93        14
      joshua_tree       1.00      0.77      0.87        22

      avg / total       0.82      0.75      0.76      1709

0.748976009362

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




CAL256 : Contrastive best :  savemodel/CAL256/netconv34.pth (checked till 74)



                               precision    recall  f1-score   support

                        other       0.85      0.85      0.85        20
              094.guitar-pick       0.88      0.44      0.59        34
      063.electric-guitar-101       0.24      0.46      0.32        26
           156.paper-shredder       0.20      0.48      0.28        23
                  096.hammock       0.52      0.50      0.51        30
             235.umbrella-101       0.56      0.62      0.59        29
              246.wine-bottle       0.25      0.09      0.13        22
             174.rotary-phone       0.14      0.33      0.20        21
              057.dolphin-101       0.25      0.23      0.24        26
            226.traffic-light       0.24      0.42      0.31        19
                     117.ipod       0.08      0.09      0.09        23
                    125.knife       0.67      0.48      0.56        21
             062.eiffel-tower       0.56      0.26      0.36        19
           202.steering-wheel       0.83      0.59      0.69        32
               053.desk-globe       0.62      0.62      0.62        29
                   214.teepee       0.20      0.32      0.24        19
                 144.minotaur       0.37      0.48      0.42        29
             217.tennis-court       0.42      0.77      0.54        13
                  084.giraffe       0.07      0.09      0.08        22
                    028.camel       0.21      0.38      0.27        21
             054.diamond-ring       0.54      0.54      0.54        26
                  079.frisbee       0.70      0.58      0.64        24
                240.watch-101       0.39      0.41      0.40        27
           112.human-skeleton       1.00      0.86      0.92        21
               072.fire-truck       0.65      0.63      0.64        27
                095.hamburger       0.60      0.67      0.63        18
       086.golden-gate-bridge       0.25      0.17      0.20        29
                024.butterfly       0.58      0.75      0.65        24
                   212.teapot       0.33      0.47      0.39        17
              228.triceratops       0.25      0.11      0.15        28
                111.house-fly       0.40      0.35      0.38        17
                134.llama-101       0.62      0.40      0.48        20
             121.kangaroo-101       0.29      0.23      0.26        22
                  220.toaster       0.25      0.26      0.26        23
                  210.syringe       0.21      0.38      0.27        21
                   231.tripod       0.10      0.16      0.12        25
             163.playing-card       0.14      0.21      0.17        24
                 013.birdbath       0.25      0.50      0.34        26
                  236.unicorn       0.62      0.62      0.62        16
                   082.galaxy       0.27      0.44      0.33        18
                     207.swan       0.67      0.44      0.53        27
                023.bulldozer       0.88      0.64      0.74        22
                     256.toad       0.60      0.52      0.56        29
                011.billiards       0.42      0.42      0.42        26
              132.light-house       0.36      0.40      0.38        20
                  183.sextant       0.45      0.63      0.52        27
                  170.rainbow       0.95      0.91      0.93        23
               178.school-bus       0.29      0.19      0.23        26
                 229.tricycle       0.48      0.46      0.47        26
               015.bonsai-101       0.62      0.62      0.62        34
                078.fried-egg       0.11      0.06      0.07        18
                 245.windmill       0.70      0.58      0.64        24
                 083.gas-pump       0.27      0.37      0.31        27
                     026.cake       0.76      0.59      0.67        22
           175.roulette-wheel       0.30      0.44      0.36        27
            160.pez-dispenser       0.12      0.12      0.12        16
              069.fighter-jet       0.46      0.68      0.55        19
           005.baseball-glove       0.70      0.27      0.39        26
             064.elephant-101       0.36      0.48      0.41        21
                040.cockroach       0.92      0.92      0.92        24
               213.teddy-bear       0.45      0.28      0.34        18
         208.swiss-army-knife       0.73      0.73      0.73        22
                     098.harp       0.40      0.40      0.40        25
                     009.bear       1.00      0.77      0.87        30
                164.porcupine       0.71      0.50      0.59        24
                 103.hibiscus       0.08      0.18      0.11        28
                  148.mussels       0.67      0.47      0.55        17
                 114.ibis-101       0.59      0.50      0.54        34
                    038.chimp       0.84      0.67      0.74        24
               022.buddha-101       1.00      0.82      0.90        17
                    250.zebra       0.55      0.48      0.52        33
            002.american-flag       0.82      0.35      0.49        26
                142.microwave       0.67      0.47      0.55        30
            146.mountain-bike       0.35      0.26      0.30        23
                  223.top-hat       0.18      0.15      0.16        27
                155.paperclip       0.04      0.03      0.04        29
               074.flashlight       0.33      0.35      0.34        17
                   176.saddle       0.57      0.55      0.56        22
                  032.cartman       0.23      0.16      0.19        19
                   042.coffin       0.89      0.64      0.74        25
             172.revolver-101       1.00      0.89      0.94        27
              077.french-horn       0.15      0.18      0.16        17
             162.picnic-table       0.33      0.19      0.24        27
                    089.goose       0.58      0.41      0.48        17
                  234.tweezer       0.50      0.60      0.55        15
                  151.ostrich       0.52      0.44      0.48        25
                 003.backpack       0.25      0.19      0.22        21
                227.treadmill       0.09      0.11      0.10        37
                  135.mailbox       0.48      0.59      0.53        17
                    194.socks       0.21      0.33      0.25        21
           166.praying-mantis       0.86      0.60      0.71        30
              099.harpsichord       0.48      0.59      0.53        22
                 136.mandolin       0.13      0.17      0.15        18
                    173.rifle       0.93      0.58      0.72        24
                123.ketch-101       0.00      0.00      0.00        21
                     085.goat       0.90      0.70      0.79        27
             201.starfish-101       1.00      0.55      0.71        22
            230.trilobite-101       0.28      0.29      0.29        17
             019.boxing-glove       0.06      0.05      0.06        19
               211.tambourine       0.23      0.19      0.20        27
           106.horseshoe-crab       0.20      0.20      0.20        15
          006.basketball-hoop       0.29      0.50      0.36        16
                    048.conch       0.81      0.62      0.70        21
            204.sunflower-101       0.62      0.48      0.54        21
                  090.gorilla       0.90      0.78      0.84        23
               225.tower-pisa       0.07      0.33      0.12        21
                  257.clutter       0.05      0.06      0.05        16
                   025.cactus       0.09      0.12      0.10        25
                 147.mushroom       1.00      0.35      0.52        17
               041.coffee-mug       0.84      0.64      0.73        25
               067.eyeglasses       0.77      0.83      0.80        24
                     137.mars       0.50      0.24      0.32        25
             255.tennis-shoes       0.17      0.22      0.19        23
             004.baseball-bat       0.21      0.27      0.24        11
        045.computer-keyboard       0.14      0.29      0.19        21
                    105.horse       0.27      0.25      0.26        24
                139.megaphone       0.59      0.46      0.52        28
         046.computer-monitor       0.58      0.35      0.44        20
                 016.boom-box       0.19      0.26      0.22        23
                    190.snake       0.50      0.24      0.32        21
                196.spaghetti       0.35      0.21      0.26        29
          239.washing-machine       0.36      0.45      0.40        22
                 087.goldfish       0.44      0.65      0.52        17
              037.chess-board       0.50      0.41      0.45        27
                 248.yarmulke       0.38      0.19      0.26        26
                     043.coin       0.80      0.41      0.55        29
               021.breadmaker       0.40      0.38      0.39        32
              093.grasshopper       0.00      0.00      0.00        16
               185.skateboard       0.00      0.00      0.00        23
              180.screwdriver       0.57      0.31      0.40        26
                       033.cd       0.77      0.45      0.57        22
                     068.fern       0.45      0.50      0.47        18
                  109.hot-tub       0.44      0.88      0.59        17
                   092.grapes       0.36      0.17      0.24        23
           102.helicopter-101       0.05      0.05      0.05        22
                    209.sword       0.24      0.25      0.24        16
               141.microscope       0.33      0.29      0.31        14
                 205.superman       0.09      0.07      0.08        28
                247.xylophone       0.95      0.90      0.93        21
             129.leopards-101       0.15      0.28      0.20        18
                 138.mattress       0.37      0.27      0.31        26
                    249.yo-yo       0.75      0.36      0.49        25
                      237.vcr       0.12      0.31      0.17        13
                    206.sushi       0.47      0.30      0.37        23
             179.scorpion-101       0.03      0.03      0.03        33
                061.dumb-bell       0.47      0.31      0.37        26
               197.speed-boat       0.27      0.40      0.32        25
              101.head-phones       0.33      0.30      0.32        23
                   116.iguana       0.82      0.35      0.49        26
            050.covered-wagon       1.00      0.41      0.58        27
              184.sheet-music       0.18      0.29      0.22        21
                   029.cannon       0.55      0.80      0.65        15
            100.hawksbill-101       0.40      0.32      0.35        19
                   221.tomato       0.29      0.61      0.39        23
                  232.t-shirt       0.48      0.69      0.56        16
                 157.pci-card       0.47      0.36      0.41        25
                 052.crab-101       0.70      0.25      0.37        28
                073.fireworks       0.40      0.55      0.46        11
                154.palm-tree       0.80      0.75      0.77        32
           145.motorbikes-101       0.22      0.24      0.23        21
                  150.octopus       0.50      0.44      0.47        25
                049.cormorant       0.34      0.33      0.34        30
             119.jesus-christ       0.23      0.24      0.24        25
              018.bowling-pin       0.36      0.32      0.34        25
                  167.pyramid       0.45      0.60      0.52        25
        070.fire-extinguisher       0.56      0.68      0.61        28
              216.tennis-ball       0.71      0.65      0.68        23
               127.laptop-101       0.88      0.74      0.80        19
               012.binoculars       1.00      0.65      0.79        26
                110.hourglass       0.07      0.15      0.09        20
                   159.people       0.35      0.27      0.30        26
          169.radio-telescope       0.76      0.43      0.55        30
                034.centipede       0.69      0.35      0.46        26
                222.tombstone       0.85      0.61      0.71        28
                  143.minaret       0.06      0.08      0.07        25
                 058.doorknob       0.23      0.28      0.25        25
                088.golf-ball       0.36      0.19      0.25        21
             224.touring-bike       1.00      0.37      0.54        27
                020.brain-101       0.60      0.33      0.43        18
               081.frying-pan       0.25      0.11      0.15        19
               219.theodolite       0.09      0.09      0.09        22
                     055.dice       0.48      0.61      0.54        18
            215.telephone-box       0.43      0.19      0.26        16
                    128.lathe       0.08      0.15      0.10        20
                   198.spider       0.25      0.29      0.27        17
             171.refrigerator       0.20      0.25      0.22        36
                131.lightbulb       0.25      0.24      0.24        21
                 031.car-tire       0.56      0.62      0.59        32
                133.lightning       0.11      0.19      0.14        16
               242.watermelon       0.78      0.88      0.82        24
                    044.comet       0.44      0.30      0.36        27
                 203.stirrups       0.87      0.68      0.76        19
                     001.ak47       0.52      0.44      0.48        25
                  149.necktie       0.46      0.48      0.47        27
           047.computer-mouse       0.88      0.50      0.64        28
            130.license-plate       0.50      0.21      0.30        19
                097.harmonica       0.63      0.60      0.62        20
          238.video-projector       0.30      0.14      0.19        22
                    014.blimp       0.23      0.10      0.14        29
                  108.hot-dog       0.26      0.53      0.35        15
                     118.iris       0.63      0.48      0.55        25
               027.calculator       0.30      0.27      0.29        26
                241.waterfall       0.53      0.34      0.42        29
               039.chopsticks       0.28      0.29      0.28        28
                     060.duck       0.33      0.28      0.30        25
                    030.canoe       0.00      0.00      0.00        33
           059.drinking-straw       0.57      0.48      0.52        25
               035.cereal-box       0.75      0.43      0.55        21
                 066.ewer-101       0.39      0.50      0.44        22
             017.bowling-ball       0.03      0.14      0.05        14
                   126.ladder       0.70      0.35      0.47        20
          107.hot-air-balloon       1.00      0.58      0.73        19
          091.grand-piano-101       0.68      0.55      0.61        31
               051.cowboy-hat       0.17      0.32      0.22        22
                120.joy-stick       0.15      0.23      0.18        13
                  191.sneaker       0.41      0.30      0.35        30
                      056.dog       0.75      0.41      0.53        37
            104.homer-simpson       0.29      0.18      0.22        22
                    122.kayak       0.96      1.00      0.98        22
           253.faces-easy-101       0.58      0.48      0.52        23
           036.chandelier-101       0.33      0.15      0.21        26
               153.palm-pilot       0.19      0.39      0.25        18
                      007.bat       0.32      0.33      0.32        18
              193.soccer-ball       0.94      0.71      0.81        21
182.self-propelled-lawn-mower       0.85      0.77      0.81        22
                    186.skunk       0.73      0.35      0.48        31
                      065.elk       0.50      0.38      0.43        24
             071.fire-hydrant       0.20      0.44      0.27        16
                    199.spoon       0.64      0.45      0.53        20
                   177.saturn       0.73      0.80      0.76        20
          076.football-helmet       0.20      0.19      0.20        31
               188.smokestack       0.88      0.68      0.76        31
                 010.beer-mug       0.54      0.54      0.54        28
                      152.owl       0.03      0.04      0.03        26
                 195.soda-can       0.06      0.09      0.07        22
                  008.bathtub       0.75      0.65      0.70        23
                   181.segway       0.25      0.10      0.14        21
               192.snowmobile       0.64      0.32      0.42        22
              113.hummingbird       0.86      0.90      0.88        21
              161.photocopier       0.47      0.23      0.30        31
             124.killer-whale       0.47      0.39      0.43        23
                     080.frog       0.81      0.75      0.78        28
            251.airplanes-101       0.22      0.33      0.27        24
                     165.pram       0.15      0.17      0.16        30
              244.wheelbarrow       0.15      0.33      0.20        18
              075.floppy-disk       0.16      0.25      0.20        16
              233.tuning-fork       0.06      0.03      0.04        29
                    189.snail       0.31      0.19      0.24        21
               187.skyscraper       0.64      0.33      0.44        27
                254.greyhound       0.17      0.14      0.15        28
             243.welding-mask       0.42      0.26      0.32        19
           115.ice-cream-cone       0.42      0.36      0.39        22
                  158.penguin       0.21      0.15      0.18        20
            200.stained-glass       0.27      0.50      0.35        14
              140.menorah-101       0.57      0.67      0.62        18
            218.tennis-racket       0.84      0.73      0.78        22

                  avg / total       0.47      0.40      0.42      5942

0.399865365197













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