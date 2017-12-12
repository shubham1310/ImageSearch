In this project, I am working on Oxford dataset and doing image retrieval. I am using a Siamese kind of network. I am trying different kind of loss.
The starter code was taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch and changed as needed

CAL101 : Contrastive best :  savemodel/CAL101/netconv99.pth (checked from 108 till 91)

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

CAL101 : Dot Product best :  savemodel/CAL101dot/netconv68.pth (checked till 28)

                   precision    recall  f1-score   support

Using 6 number of neighbours
                   precision    recall  f1-score   support

            other       1.00      0.85      0.92        20
       Faces_easy       1.00      0.92      0.96        12
            pizza       1.00      0.45      0.62        20
          gerenuk       0.75      0.75      0.75        16
            llama       0.95      1.00      0.97        19
           camera       0.74      0.87      0.80        23
            lotus       1.00      0.70      0.82        10
        euphonium       0.65      0.83      0.73        18
              ant       1.00      1.00      1.00        17
          rooster       0.57      0.89      0.70         9
           mayfly       1.00      0.81      0.90        16
            panda       0.73      0.61      0.67        18
          lobster       0.71      0.75      0.73        20
         schooner       0.64      1.00      0.78        14
       chandelier       0.95      0.87      0.91        23
          menorah       1.00      0.72      0.84        18
        saxophone       0.89      0.93      0.91        27
         flamingo       0.88      1.00      0.93        14
   crocodile_head       1.00      1.00      1.00        24
       wheelchair       0.68      0.72      0.70        18
            chair       1.00      1.00      1.00        17
        dragonfly       1.00      1.00      1.00        17
        binocular       0.18      0.77      0.29        22
BACKGROUND_Google       0.86      0.38      0.52        16
          octopus       0.29      0.67      0.41        15
             crab       0.75      0.25      0.38        12
        crocodile       1.00      0.78      0.88        18
         mandolin       1.00      0.88      0.94        17
      dollar_bill       1.00      0.95      0.98        22
      stegosaurus       1.00      1.00      1.00        21
     inline_skate       0.88      0.93      0.90        15
           bonsai       1.00      0.94      0.97        16
            okapi       0.71      0.62      0.67        16
    windsor_chair       0.81      0.87      0.84        15
      cougar_body       0.72      0.93      0.81        14
       helicopter       0.95      1.00      0.97        19
         hedgehog       1.00      0.95      0.97        19
        trilobite       0.93      1.00      0.97        14
        cellphone       1.00      0.80      0.89        20
          dolphin       0.87      0.81      0.84        16
         elephant       0.89      1.00      0.94        17
              emu       0.85      0.69      0.76        16
        accordion       1.00      0.95      0.98        21
             ewer       1.00      0.67      0.80        15
        sunflower       0.94      0.89      0.92        19
             tick       0.80      0.92      0.86        13
            brain       0.83      0.67      0.74        15
             ibis       0.74      1.00      0.85        14
           pagoda       1.00      0.91      0.95        22
    flamingo_head       0.94      1.00      0.97        16
         wild_cat       0.94      0.79      0.86        19
           beaver       0.84      1.00      0.91        16
            Faces       1.00      0.80      0.89        15
      soccer_ball       1.00      0.93      0.96        14
             lamp       0.71      0.94      0.81        16
         yin_yang       1.00      1.00      1.00        17
       gramophone       0.75      0.75      0.75        24
           anchor       0.75      0.67      0.71        18
         kangaroo       1.00      0.89      0.94         9
         garfield       1.00      0.35      0.52        20
           wrench       1.00      0.93      0.96        14
           snoopy       1.00      0.56      0.72        16
           cannon       1.00      1.00      1.00        13
         scissors       1.00      0.88      0.93        16
        airplanes       0.38      0.26      0.31        23
         crayfish       1.00      0.84      0.91        19
      ceiling_fan       0.60      0.50      0.55        12
      water_lilly       1.00      0.87      0.93        15
        dalmatian       0.73      0.55      0.63        20
     brontosaurus       1.00      0.94      0.97        16
              cup       1.00      0.65      0.79        23
        butterfly       1.00      0.75      0.86         8
  electric_guitar       1.00      1.00      1.00        12
      grand_piano       1.00      0.89      0.94        18
            rhino       0.96      0.96      0.96        24
          minaret       0.70      0.76      0.73        21
      cougar_face       0.69      0.55      0.61        20
            ketch       0.73      0.92      0.81        12
        sea_horse       1.00      1.00      1.00        19
        metronome       1.00      0.83      0.91        18
           pigeon       1.00      1.00      1.00        14
       Motorbikes       0.96      0.92      0.94        25
            watch       1.00      0.92      0.96        12
        headphone       0.81      1.00      0.89        17
        stop_sign       1.00      0.71      0.83        17
          stapler       1.00      1.00      1.00        17
       strawberry       1.00      1.00      1.00        17
         revolver       0.93      0.87      0.90        15
          pyramid       1.00      0.86      0.92        14
           laptop       0.91      1.00      0.95        10
           barrel       0.40      0.47      0.43        17
         scorpion       1.00      0.93      0.97        15
           buddha       1.00      0.76      0.87        17
         platypus       0.82      1.00      0.90        18
        hawksbill       0.85      0.92      0.88        12
         Leopards       1.00      0.85      0.92        13
         nautilus       0.68      1.00      0.81        15
             bass       1.00      0.87      0.93        23
         starfish       0.92      0.69      0.79        16
         umbrella       0.94      1.00      0.97        15
      joshua_tree       0.90      1.00      0.95        18

      avg / total       0.88      0.83      0.84      1709

0.832650672908



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


CAL256 : dot product loss best :  savemodel/CAL256dot/netconv63.pth (checked from 65 to 58)

Using 9 number of neighbours
                               precision    recall  f1-score   support

                        other       0.60      0.88      0.71        24
              094.guitar-pick       0.69      0.71      0.70        28
      063.electric-guitar-101       0.41      0.47      0.44        19
           156.paper-shredder       0.21      0.21      0.21        24
                  096.hammock       1.00      0.41      0.58        27
             235.umbrella-101       0.48      0.56      0.52        25
              246.wine-bottle       0.48      0.62      0.54        21
             174.rotary-phone       0.45      0.46      0.46        28
              057.dolphin-101       0.50      0.29      0.36        21
            226.traffic-light       0.86      0.83      0.84        23
                     117.ipod       0.12      0.12      0.12        24
                    125.knife       0.79      0.59      0.68        32
             062.eiffel-tower       0.57      0.53      0.55        30
           202.steering-wheel       0.80      0.80      0.80        20
               053.desk-globe       0.38      0.53      0.44        17
                   214.teepee       0.56      0.36      0.43        14
                 144.minotaur       0.42      0.74      0.53        38
             217.tennis-court       0.73      0.41      0.52        27
                  084.giraffe       0.22      0.33      0.26        33
                    028.camel       0.58      0.82      0.68        22
             054.diamond-ring       0.60      0.55      0.57        33
                  079.frisbee       0.79      0.77      0.78        30
                240.watch-101       0.44      0.64      0.52        22
           112.human-skeleton       0.75      0.88      0.81        17
               072.fire-truck       0.71      0.67      0.69        15
                095.hamburger       0.56      0.83      0.67        24
       086.golden-gate-bridge       0.86      0.25      0.39        24
                024.butterfly       0.82      0.56      0.67        16
                   212.teapot       0.50      0.40      0.44        15
              228.triceratops       0.37      0.54      0.43        28
                111.house-fly       0.38      0.35      0.36        23
                134.llama-101       0.22      0.62      0.33        16
             121.kangaroo-101       0.06      0.05      0.05        19
                  220.toaster       0.28      0.38      0.32        24
                  210.syringe       0.70      0.33      0.45        21
                   231.tripod       0.26      0.33      0.29        21
             163.playing-card       0.29      0.18      0.22        22
                 013.birdbath       0.40      0.29      0.34        41
                  236.unicorn       0.54      0.52      0.53        29
                   082.galaxy       0.73      0.46      0.56        24
                     207.swan       0.45      0.78      0.57        27
                023.bulldozer       0.31      0.61      0.41        18
                     256.toad       0.54      0.50      0.52        14
                011.billiards       0.53      0.53      0.53        17
              132.light-house       0.48      0.71      0.58        21
                  183.sextant       0.30      0.56      0.39        16
                  170.rainbow       0.96      0.90      0.93        30
               178.school-bus       0.56      0.15      0.23        34
                 229.tricycle       0.74      0.54      0.62        26
               015.bonsai-101       0.36      0.40      0.38        25
                078.fried-egg       0.40      0.22      0.29        18
                 245.windmill       0.30      0.27      0.29        26
                 083.gas-pump       0.25      0.11      0.15        27
                     026.cake       0.59      0.57      0.58        23
           175.roulette-wheel       0.75      0.60      0.67        20
            160.pez-dispenser       0.70      0.29      0.41        24
              069.fighter-jet       0.37      0.35      0.36        20
           005.baseball-glove       0.45      0.83      0.59        18
             064.elephant-101       0.40      0.18      0.25        22
                040.cockroach       1.00      0.71      0.83        14
               213.teddy-bear       0.28      0.42      0.33        24
         208.swiss-army-knife       1.00      0.63      0.78        30
                     098.harp       0.40      0.12      0.19        16
                     009.bear       0.85      0.59      0.69        29
                164.porcupine       0.57      0.74      0.65        27
                 103.hibiscus       0.16      0.29      0.20        21
                  148.mussels       0.44      0.65      0.53        23
                 114.ibis-101       1.00      0.12      0.21        25
                    038.chimp       0.65      0.68      0.67        22
               022.buddha-101       1.00      0.88      0.93        24
                    250.zebra       0.45      0.54      0.49        24
            002.american-flag       0.56      0.40      0.47        25
                142.microwave       0.39      0.28      0.33        25
            146.mountain-bike       0.47      0.88      0.61        17
                  223.top-hat       0.22      0.17      0.20        23
                155.paperclip       0.53      0.32      0.40        25
               074.flashlight       0.80      0.70      0.74        23
                   176.saddle       0.68      0.83      0.75        18
                  032.cartman       0.20      0.37      0.26        19
                   042.coffin       0.56      0.68      0.61        37
             172.revolver-101       0.95      0.90      0.93        21
              077.french-horn       0.14      0.14      0.14        22
             162.picnic-table       0.22      0.17      0.19        30
                    089.goose       0.50      0.88      0.64        16
                  234.tweezer       0.60      0.68      0.64        22
                  151.ostrich       0.91      0.74      0.82        27
                 003.backpack       0.79      0.71      0.75        31
                227.treadmill       0.22      0.07      0.11        27
                  135.mailbox       0.66      0.86      0.75        36
                    194.socks       0.50      0.38      0.43        26
           166.praying-mantis       0.62      0.48      0.54        21
              099.harpsichord       0.47      0.44      0.46        18
                 136.mandolin       0.15      0.17      0.16        23
                    173.rifle       0.82      0.82      0.82        22
                123.ketch-101       0.08      0.07      0.07        30
                     085.goat       0.61      0.79      0.69        14
             201.starfish-101       0.96      0.90      0.93        30
            230.trilobite-101       0.74      0.61      0.67        28
             019.boxing-glove       0.46      0.27      0.34        22
               211.tambourine       0.33      0.27      0.30        26
           106.horseshoe-crab       0.41      0.41      0.41        22
          006.basketball-hoop       0.58      0.47      0.52        30
                    048.conch       0.89      0.94      0.92        18
            204.sunflower-101       0.36      0.59      0.45        22
                  090.gorilla       0.53      0.53      0.53        17
               225.tower-pisa       0.12      0.50      0.20        24
                  257.clutter       0.12      0.05      0.07        19
                   025.cactus       0.55      0.20      0.29        30
                 147.mushroom       0.65      0.58      0.61        26
               041.coffee-mug       0.78      0.82      0.80        22
               067.eyeglasses       0.42      0.89      0.58        19
                     137.mars       0.38      0.50      0.43        24
             255.tennis-shoes       0.11      0.09      0.10        22
             004.baseball-bat       0.61      0.64      0.62        22
        045.computer-keyboard       0.50      0.34      0.41        29
                    105.horse       0.35      0.61      0.45        18
                139.megaphone       0.57      0.20      0.30        20
         046.computer-monitor       0.64      0.57      0.60        28
                 016.boom-box       0.47      0.27      0.34        30
                    190.snake       0.30      0.41      0.35        17
                196.spaghetti       0.36      0.64      0.46        14
          239.washing-machine       0.83      0.53      0.65        19
                 087.goldfish       0.53      0.78      0.63        27
              037.chess-board       0.60      0.57      0.59        21
                 248.yarmulke       0.31      0.25      0.28        20
                     043.coin       0.73      0.46      0.56        24
               021.breadmaker       0.68      0.63      0.66        30
              093.grasshopper       0.04      0.04      0.04        23
               185.skateboard       0.23      0.22      0.23        27
              180.screwdriver       0.35      0.53      0.42        17
                       033.cd       0.45      0.60      0.51        30
                     068.fern       0.73      0.52      0.61        21
                  109.hot-tub       0.56      0.83      0.67        18
                   092.grapes       0.52      0.50      0.51        22
           102.helicopter-101       0.00      0.00      0.00        19
                    209.sword       0.20      0.17      0.18        24
               141.microscope       0.33      0.16      0.21        19
                 205.superman       0.53      0.32      0.40        25
                247.xylophone       0.75      1.00      0.86        33
             129.leopards-101       0.32      0.52      0.39        27
                 138.mattress       0.26      0.23      0.24        22
                    249.yo-yo       0.69      0.75      0.72        32
                      237.vcr       0.58      0.39      0.47        18
                    206.sushi       0.33      0.41      0.37        22
             179.scorpion-101       0.72      0.60      0.65        30
                061.dumb-bell       0.52      0.35      0.42        31
               197.speed-boat       0.53      0.45      0.49        20
              101.head-phones       0.35      0.25      0.29        24
                   116.iguana       0.25      0.32      0.28        22
            050.covered-wagon       0.88      1.00      0.93        28
              184.sheet-music       0.60      0.29      0.39        21
                   029.cannon       0.67      0.73      0.70        22
            100.hawksbill-101       0.74      0.74      0.74        23
                   221.tomato       0.80      0.73      0.76        22
                  232.t-shirt       0.64      0.72      0.68        25
                 157.pci-card       0.36      0.53      0.43        19
                 052.crab-101       0.56      0.56      0.56        27
                073.fireworks       0.62      0.48      0.54        27
                154.palm-tree       0.75      0.60      0.67        20
           145.motorbikes-101       0.25      0.17      0.21        23
                  150.octopus       0.74      0.85      0.79        27
                049.cormorant       0.28      0.65      0.39        17
             119.jesus-christ       0.50      0.32      0.39        25
              018.bowling-pin       0.31      0.67      0.43        15
                  167.pyramid       0.58      0.38      0.45        40
        070.fire-extinguisher       0.37      0.62      0.46        21
              216.tennis-ball       0.62      0.84      0.71        25
               127.laptop-101       0.90      0.45      0.60        20
               012.binoculars       0.73      0.62      0.67        26
                110.hourglass       0.29      0.48      0.36        21
                   159.people       0.46      0.48      0.47        25
          169.radio-telescope       1.00      0.59      0.74        17
                034.centipede       0.83      0.68      0.75        22
                222.tombstone       0.67      0.50      0.57        20
                  143.minaret       0.36      0.22      0.27        23
                 058.doorknob       0.70      0.29      0.41        24
                088.golf-ball       0.50      0.59      0.54        17
             224.touring-bike       0.70      0.86      0.78        22
                020.brain-101       0.64      0.57      0.60        28
               081.frying-pan       0.58      0.37      0.45        19
               219.theodolite       0.57      0.33      0.42        24
                     055.dice       0.59      0.77      0.67        35
            215.telephone-box       0.31      0.38      0.34        21
                    128.lathe       0.10      0.15      0.12        13
                   198.spider       0.31      0.36      0.33        33
             171.refrigerator       0.64      0.70      0.67        23
                131.lightbulb       0.65      0.74      0.69        23
                 031.car-tire       0.47      0.77      0.58        26
                133.lightning       0.22      0.18      0.20        22
               242.watermelon       0.48      0.59      0.53        27
                    044.comet       0.65      0.52      0.58        21
                 203.stirrups       0.48      0.39      0.43        31
                     001.ak47       0.67      0.67      0.67        27
                  149.necktie       0.41      0.47      0.44        19
           047.computer-mouse       0.74      0.81      0.78        32
            130.license-plate       0.60      0.33      0.43        18
                097.harmonica       0.54      0.37      0.44        19
          238.video-projector       0.81      0.50      0.62        34
                    014.blimp       0.40      0.19      0.26        21
                  108.hot-dog       0.35      0.41      0.38        27
                     118.iris       0.56      0.58      0.57        24
               027.calculator       0.81      0.74      0.77        23
                241.waterfall       0.56      0.60      0.58        25
               039.chopsticks       0.47      0.36      0.41        22
                     060.duck       0.33      0.41      0.37        22
                    030.canoe       0.00      0.00      0.00        23
           059.drinking-straw       0.73      0.44      0.55        18
               035.cereal-box       0.73      0.73      0.73        22
                 066.ewer-101       0.26      0.21      0.23        24
             017.bowling-ball       0.27      0.32      0.29        22
                   126.ladder       0.81      0.76      0.79        17
          107.hot-air-balloon       0.67      0.83      0.74        24
          091.grand-piano-101       1.00      0.40      0.57        20
               051.cowboy-hat       0.65      0.39      0.49        28
                120.joy-stick       0.29      0.26      0.27        23
                  191.sneaker       0.50      0.19      0.28        21
                      056.dog       0.51      0.87      0.65        23
            104.homer-simpson       0.44      0.50      0.47        22
                    122.kayak       1.00      1.00      1.00        22
           253.faces-easy-101       0.80      0.62      0.70        13
           036.chandelier-101       0.91      0.38      0.54        26
               153.palm-pilot       0.52      0.55      0.54        20
                      007.bat       0.88      0.54      0.67        28
              193.soccer-ball       0.61      0.95      0.74        21
182.self-propelled-lawn-mower       0.79      0.66      0.72        29
                    186.skunk       0.28      0.32      0.30        25
                      065.elk       0.74      0.50      0.60        28
             071.fire-hydrant       0.32      0.32      0.32        19
                    199.spoon       0.47      0.60      0.53        15
                   177.saturn       0.58      0.65      0.61        17
          076.football-helmet       0.14      0.15      0.15        26
               188.smokestack       0.75      0.78      0.77        23
                 010.beer-mug       0.64      0.67      0.65        24
                      152.owl       0.50      0.17      0.25        24
                 195.soda-can       0.20      0.26      0.23        23
                  008.bathtub       0.36      0.59      0.45        22
                   181.segway       0.54      0.62      0.58        21
               192.snowmobile       0.69      0.64      0.67        28
              113.hummingbird       0.35      0.47      0.40        17
              161.photocopier       0.66      0.93      0.77        27
             124.killer-whale       0.25      0.11      0.15        19
                     080.frog       0.57      0.80      0.67        20
            251.airplanes-101       0.68      0.54      0.60        28
                     165.pram       0.31      0.38      0.34        21
              244.wheelbarrow       0.31      0.48      0.38        25
              075.floppy-disk       0.15      0.15      0.15        20
              233.tuning-fork       0.25      0.12      0.17        16
                    189.snail       0.56      0.45      0.50        11
               187.skyscraper       0.43      0.56      0.49        16
                254.greyhound       0.39      0.58      0.47        19
             243.welding-mask       0.33      0.23      0.27        30
           115.ice-cream-cone       0.50      0.13      0.21        23
                  158.penguin       1.00      0.38      0.56        26
            200.stained-glass       0.73      0.70      0.71        23
              140.menorah-101       1.00      0.94      0.97        16
            218.tennis-racket       0.53      0.82      0.64        11

                  avg / total       0.53      0.50      0.50      5942

0.503702457085



CAL256 : neural loss best : savemodel/CAL256neural/netconv65.pth

Using 3 number of neighbours
                               precision    recall  f1-score   support

                        other       0.14      0.43      0.21        30
              094.guitar-pick       0.03      0.04      0.04        23
      063.electric-guitar-101       0.09      0.21      0.12        28
           156.paper-shredder       0.00      0.00      0.00        19
                  096.hammock       0.12      0.24      0.16        21
             235.umbrella-101       0.08      0.08      0.08        13
              246.wine-bottle       0.00      0.00      0.00        21
             174.rotary-phone       0.00      0.00      0.00        20
              057.dolphin-101       0.00      0.00      0.00        20
            226.traffic-light       0.03      0.07      0.04        14
                     117.ipod       0.00      0.00      0.00        20
                    125.knife       0.43      0.12      0.19        24
             062.eiffel-tower       0.00      0.00      0.00        24
           202.steering-wheel       0.07      0.14      0.09        21
               053.desk-globe       0.17      0.15      0.16        20
                   214.teepee       0.00      0.00      0.00        19
                 144.minotaur       0.50      0.11      0.18        18
             217.tennis-court       0.07      0.07      0.07        27
                  084.giraffe       0.00      0.00      0.00        25
                    028.camel       0.11      0.41      0.18        29
             054.diamond-ring       0.10      0.26      0.14        23
                  079.frisbee       0.28      0.72      0.40        25
                240.watch-101       0.05      0.15      0.08        20
           112.human-skeleton       0.45      0.23      0.30        22
               072.fire-truck       0.00      0.00      0.00        17
                095.hamburger       0.13      0.14      0.14        21
       086.golden-gate-bridge       0.07      0.06      0.07        31
                024.butterfly       0.07      0.08      0.07        26
                   212.teapot       0.09      0.06      0.07        17
              228.triceratops       0.19      0.27      0.22        33
                111.house-fly       0.38      0.15      0.21        20
                134.llama-101       0.09      0.03      0.05        29
             121.kangaroo-101       0.03      0.05      0.04        19
                  220.toaster       0.00      0.00      0.00        22
                  210.syringe       0.21      0.10      0.13        31
                   231.tripod       0.00      0.00      0.00        28
             163.playing-card       0.00      0.00      0.00        20
                 013.birdbath       0.00      0.00      0.00        23
                  236.unicorn       0.11      0.31      0.16        13
                   082.galaxy       0.00      0.00      0.00        27
                     207.swan       0.10      0.16      0.12        19
                023.bulldozer       0.04      0.04      0.04        26
                     256.toad       0.00      0.00      0.00        25
                011.billiards       0.00      0.00      0.00        25
              132.light-house       0.14      0.24      0.18        25
                  183.sextant       0.19      0.28      0.22        18
                  170.rainbow       0.54      0.33      0.41        21
               178.school-bus       0.08      0.11      0.09        27
                 229.tricycle       0.06      0.11      0.07        28
               015.bonsai-101       0.00      0.00      0.00        31
                078.fried-egg       0.00      0.00      0.00        23
                 245.windmill       0.00      0.00      0.00        23
                 083.gas-pump       0.00      0.00      0.00        18
                     026.cake       0.35      0.41      0.38        17
           175.roulette-wheel       0.00      0.00      0.00        22
            160.pez-dispenser       0.05      0.04      0.04        24
              069.fighter-jet       0.01      0.05      0.02        19
           005.baseball-glove       0.10      0.14      0.12        21
             064.elephant-101       0.03      0.04      0.04        23
                040.cockroach       0.16      0.24      0.19        25
               213.teddy-bear       0.02      0.08      0.04        24
         208.swiss-army-knife       0.50      0.16      0.24        19
                     098.harp       0.00      0.00      0.00        23
                     009.bear       0.00      0.00      0.00        20
                164.porcupine       0.06      0.07      0.06        15
                 103.hibiscus       0.00      0.00      0.00        20
                  148.mussels       0.08      0.05      0.06        19
                 114.ibis-101       0.67      0.10      0.17        21
                    038.chimp       0.07      0.11      0.09        18
               022.buddha-101       0.08      0.04      0.05        26
                    250.zebra       0.19      0.22      0.20        23
            002.american-flag       0.00      0.00      0.00        18
                142.microwave       0.08      0.11      0.09        19
            146.mountain-bike       0.08      0.19      0.11        27
                  223.top-hat       0.00      0.00      0.00        27
                155.paperclip       0.00      0.00      0.00        32
               074.flashlight       0.04      0.04      0.04        23
                   176.saddle       0.04      0.08      0.05        24
                  032.cartman       0.00      0.00      0.00        17
                   042.coffin       0.24      0.30      0.26        27
             172.revolver-101       0.44      0.46      0.45        26
              077.french-horn       0.00      0.00      0.00        26
             162.picnic-table       0.00      0.00      0.00        18
                    089.goose       0.05      0.16      0.08        19
                  234.tweezer       0.00      0.00      0.00        26
                  151.ostrich       0.01      0.06      0.02        17
                 003.backpack       0.00      0.00      0.00        26
                227.treadmill       0.08      0.08      0.08        26
                  135.mailbox       0.00      0.00      0.00        34
                    194.socks       0.00      0.00      0.00        18
           166.praying-mantis       0.33      0.08      0.13        25
              099.harpsichord       0.00      0.00      0.00        23
                 136.mandolin       0.00      0.00      0.00        17
                    173.rifle       0.23      0.22      0.22        23
                123.ketch-101       0.00      0.00      0.00        23
                     085.goat       0.09      0.18      0.12        34
             201.starfish-101       0.07      0.21      0.11        19
            230.trilobite-101       0.07      0.11      0.09        28
             019.boxing-glove       0.00      0.00      0.00        22
               211.tambourine       0.00      0.00      0.00        28
           106.horseshoe-crab       0.09      0.17      0.12        18
          006.basketball-hoop       0.00      0.00      0.00        23
                    048.conch       0.17      0.18      0.18        22
            204.sunflower-101       0.10      0.04      0.06        24
                  090.gorilla       0.32      0.24      0.27        29
               225.tower-pisa       1.00      0.05      0.09        22
                  257.clutter       0.00      0.00      0.00        21
                   025.cactus       0.00      0.00      0.00        24
                 147.mushroom       0.55      0.35      0.43        17
               041.coffee-mug       0.59      0.42      0.49        24
               067.eyeglasses       0.18      0.65      0.28        20
                     137.mars       0.00      0.00      0.00        17
             255.tennis-shoes       0.00      0.00      0.00        18
             004.baseball-bat       0.00      0.00      0.00        20
        045.computer-keyboard       0.00      0.00      0.00        22
                    105.horse       0.00      0.00      0.00        21
                139.megaphone       0.25      0.26      0.26        23
         046.computer-monitor       0.08      0.06      0.07        17
                 016.boom-box       0.00      0.00      0.00        22
                    190.snake       0.00      0.00      0.00        18
                196.spaghetti       0.00      0.00      0.00        21
          239.washing-machine       0.00      0.00      0.00        21
                 087.goldfish       0.00      0.00      0.00        33
              037.chess-board       0.10      0.28      0.15        25
                 248.yarmulke       0.10      0.04      0.06        23
                     043.coin       0.19      0.45      0.26        22
               021.breadmaker       0.00      0.00      0.00        21
              093.grasshopper       0.00      0.00      0.00        22
               185.skateboard       0.00      0.00      0.00        18
              180.screwdriver       0.03      0.04      0.03        26
                       033.cd       0.03      0.04      0.04        24
                     068.fern       0.00      0.00      0.00        23
                  109.hot-tub       0.20      0.04      0.06        28
                   092.grapes       0.04      0.04      0.04        25
           102.helicopter-101       0.00      0.00      0.00        25
                    209.sword       0.00      0.00      0.00        27
               141.microscope       0.00      0.00      0.00        20
                 205.superman       0.00      0.00      0.00        20
                247.xylophone       0.51      0.69      0.59        29
             129.leopards-101       0.00      0.00      0.00        27
                 138.mattress       0.11      0.12      0.11        17
                    249.yo-yo       1.00      0.19      0.32        21
                      237.vcr       0.14      0.08      0.10        25
                    206.sushi       0.35      0.36      0.35        25
             179.scorpion-101       0.50      0.06      0.11        31
                061.dumb-bell       0.00      0.00      0.00        21
               197.speed-boat       0.00      0.00      0.00        19
              101.head-phones       0.00      0.00      0.00        24
                   116.iguana       0.06      0.05      0.05        21
            050.covered-wagon       0.33      0.27      0.30        26
              184.sheet-music       0.00      0.00      0.00        21
                   029.cannon       0.00      0.00      0.00        13
            100.hawksbill-101       0.09      0.11      0.10        19
                   221.tomato       0.17      0.16      0.17        25
                  232.t-shirt       0.15      0.15      0.15        27
                 157.pci-card       0.00      0.00      0.00        24
                 052.crab-101       0.00      0.00      0.00        22
                073.fireworks       0.00      0.00      0.00        18
                154.palm-tree       0.23      0.70      0.34        23
           145.motorbikes-101       0.00      0.00      0.00        24
                  150.octopus       0.00      0.00      0.00        16
                049.cormorant       0.13      0.13      0.13        30
             119.jesus-christ       0.00      0.00      0.00        27
              018.bowling-pin       0.05      0.05      0.05        20
                  167.pyramid       0.33      0.14      0.19        22
        070.fire-extinguisher       0.21      0.50      0.29        24
              216.tennis-ball       0.17      0.05      0.08        19
               127.laptop-101       0.39      0.35      0.37        26
               012.binoculars       0.06      0.04      0.05        25
                110.hourglass       0.00      0.00      0.00        20
                   159.people       0.00      0.00      0.00        25
          169.radio-telescope       0.21      0.20      0.21        20
                034.centipede       0.18      0.15      0.17        26
                222.tombstone       0.50      0.05      0.08        22
                  143.minaret       0.00      0.00      0.00        24
                 058.doorknob       0.00      0.00      0.00        14
                088.golf-ball       0.29      0.18      0.22        33
             224.touring-bike       0.06      0.08      0.07        24
                020.brain-101       0.36      0.15      0.22        26
               081.frying-pan       0.16      0.16      0.16        32
               219.theodolite       0.10      0.04      0.06        26
                     055.dice       0.11      0.07      0.09        27
            215.telephone-box       0.00      0.00      0.00        20
                    128.lathe       0.00      0.00      0.00        18
                   198.spider       0.04      0.03      0.04        29
             171.refrigerator       0.17      0.11      0.13        28
                131.lightbulb       0.23      0.16      0.19        31
                 031.car-tire       0.22      0.31      0.26        29
                133.lightning       0.00      0.00      0.00        24
               242.watermelon       0.28      0.44      0.34        27
                    044.comet       0.00      0.00      0.00        24
                 203.stirrups       0.31      0.14      0.20        28
                     001.ak47       0.06      0.03      0.04        31
                  149.necktie       0.09      0.15      0.11        33
           047.computer-mouse       0.00      0.00      0.00        27
            130.license-plate       0.15      0.06      0.09        31
                097.harmonica       0.50      0.29      0.36        28
          238.video-projector       0.00      0.00      0.00        25
                    014.blimp       0.00      0.00      0.00        14
                  108.hot-dog       0.00      0.00      0.00        25
                     118.iris       0.00      0.00      0.00        21
               027.calculator       0.00      0.00      0.00        22
                241.waterfall       0.00      0.00      0.00        18
               039.chopsticks       0.00      0.00      0.00        14
                     060.duck       0.00      0.00      0.00        23
                    030.canoe       0.00      0.00      0.00        14
           059.drinking-straw       0.17      0.11      0.13        28
               035.cereal-box       0.18      0.32      0.23        22
                 066.ewer-101       0.02      0.04      0.03        26
             017.bowling-ball       0.00      0.00      0.00        20
                   126.ladder       0.00      0.00      0.00        21
          107.hot-air-balloon       0.07      0.05      0.06        19
          091.grand-piano-101       0.10      0.11      0.11        35
               051.cowboy-hat       0.08      0.04      0.05        26
                120.joy-stick       0.07      0.06      0.07        33
                  191.sneaker       0.00      0.00      0.00        20
                      056.dog       0.29      0.22      0.25        32
            104.homer-simpson       0.00      0.00      0.00        28
                    122.kayak       0.93      0.81      0.86        31
           253.faces-easy-101       0.00      0.00      0.00        27
           036.chandelier-101       0.00      0.00      0.00        27
               153.palm-pilot       0.00      0.00      0.00        24
                      007.bat       0.00      0.00      0.00        21
              193.soccer-ball       0.82      0.61      0.70        23
182.self-propelled-lawn-mower       0.40      0.15      0.22        26
                    186.skunk       0.00      0.00      0.00        28
                      065.elk       0.00      0.00      0.00        31
             071.fire-hydrant       0.00      0.00      0.00        28
                    199.spoon       0.35      0.53      0.42        30
                   177.saturn       0.00      0.00      0.00        22
          076.football-helmet       0.00      0.00      0.00        31
               188.smokestack       0.20      0.11      0.14        28
                 010.beer-mug       0.00      0.00      0.00        27
                      152.owl       0.00      0.00      0.00        11
                 195.soda-can       0.00      0.00      0.00        26
                  008.bathtub       0.00      0.00      0.00        26
                   181.segway       0.00      0.00      0.00        19
               192.snowmobile       0.00      0.00      0.00        15
              113.hummingbird       0.00      0.00      0.00        21
              161.photocopier       0.09      0.06      0.07        16
             124.killer-whale       0.00      0.00      0.00        21
                     080.frog       0.05      0.12      0.07        16
            251.airplanes-101       0.00      0.00      0.00        27
                     165.pram       0.00      0.00      0.00        14
              244.wheelbarrow       0.00      0.00      0.00        23
              075.floppy-disk       0.00      0.00      0.00        27
              233.tuning-fork       0.00      0.00      0.00        29
                    189.snail       0.00      0.00      0.00        18
               187.skyscraper       0.00      0.00      0.00        18
                254.greyhound       0.10      0.12      0.11        17
             243.welding-mask       0.00      0.00      0.00        23
           115.ice-cream-cone       0.00      0.00      0.00        21
                  158.penguin       0.00      0.00      0.00        16
            200.stained-glass       0.00      0.00      0.00        28
              140.menorah-101       0.28      0.20      0.23        25
            218.tennis-racket       0.70      0.25      0.37        28

                  avg / total       0.11      0.10      0.09      5942

0.100134634803


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