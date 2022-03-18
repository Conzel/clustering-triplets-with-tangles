def get_coco_dicts():
    knife = ["n03041632"]
    keyboard = ["n03085013", "n04505470"]
    elephant = ["n02504013", "n02504458"]
    bicycle = ["n02835271", "n03792782"]
    airplane = ["n02690373", "n03955296", "n13861050",
                "n13941806"]
    clock = ["n02708093", "n03196217", "n04548280"]
    oven = ["n03259401", "n04111414", "n04111531"]
    chair = ["n02791124", "n03376595", "n04099969",
             "n00605023", "n04429376"]
    bear = ["n02132136", "n02133161", "n02134084",
            "n02134418"]
    boat = ["n02951358", "n03344393", "n03662601",
            "n04273569", "n04612373", "n04612504"]
    cat = ["n02122878", "n02123045", "n02123159",
           "n02126465", "n02123394", "n02123597",
           "n02124075", "n02125311"]
    bottle = ["n02823428", "n03937543", "n03983396",
              "n04557648", "n04560804", "n04579145",
              "n04591713"]
    truck = ["n03345487", "n03417042", "n03770679",
             "n03796401", "n00319176", "n01016201",
             "n03930630", "n03930777", "n05061003",
             "n06547832", "n10432053", "n03977966",
             "n04461696", "n04467665"]
    car = ["n02814533", "n03100240", "n03100346",
           "n13419325", "n04285008"]
    bird = ["n01321123", "n01514859", "n01792640",
            "n07646067", "n01530575", "n01531178", "n01532829",
            "n01534433", "n01537544", "n01558993", "n01562265",
            "n01560419", "n01582220", "n10281276", "n01592084",
            "n01601694", "n01614925", "n01616318", "n01622779",
            "n01795545", "n01796340", "n01797886", "n01798484",
            "n01817953", "n01818515", "n01819313", "n01820546",
            "n01824575", "n01828970", "n01829413", "n01833805",
            "n01843065", "n01843383", "n01855032", "n01855672",
            "n07646821", "n01860187", "n02002556", "n02002724",
            "n02006656", "n02007558", "n02009229", "n02009912",
            "n02011460", "n02013706", "n02017213", "n02018207",
            "n02018795", "n02025239", "n02027492", "n02028035",
            "n02033041", "n02037110", "n02051845", "n02056570"]
    dog = ["n02085782", "n02085936", "n02086079",
           "n02086240", "n02086646", "n02086910", "n02087046",
           "n02087394", "n02088094", "n02088238", "n02088364",
           "n02088466", "n02088632", "n02089078", "n02089867",
           "n02089973", "n02090379", "n02090622", "n02090721",
           "n02091032", "n02091134", "n02091244", "n02091467",
           "n02091635", "n02091831", "n02092002", "n02092339",
           "n02093256", "n02093428", "n02093647", "n02093754",
           "n02093859", "n02093991", "n02094114", "n02094258",
           "n02094433", "n02095314", "n02095570", "n02095889",
           "n02096051", "n02096294", "n02096437", "n02096585",
           "n02097047", "n02097130", "n02097209", "n02097298",
           "n02097474", "n02097658", "n02098105", "n02098286",
           "n02099267", "n02099429", "n02099601", "n02099712",
           "n02099849", "n02100236", "n02100583", "n02100735",
           "n02100877", "n02101006", "n02101388", "n02101556",
           "n02102040", "n02102177", "n02102318", "n02102480",
           "n02102973", "n02104029", "n02104365", "n02105056",
           "n02105162", "n02105251", "n02105505", "n02105641",
           "n02105855", "n02106030", "n02106166", "n02106382",
           "n02106550", "n02106662", "n02107142", "n02107312",
           "n02107574", "n02107683", "n02107908", "n02108000",
           "n02108422", "n02108551", "n02108915", "n02109047",
           "n02109525", "n02109961", "n02110063", "n02110185",
           "n02110627", "n02110806", "n02110958", "n02111129",
           "n02111277", "n08825211", "n02111500", "n02112018",
           "n02112350", "n02112706", "n02113023", "n02113624",
           "n02113712", "n02113799", "n02113978"]

    classes = [knife, keyboard, elephant, bicycle, airplane, clock,
               oven, chair, bear, boat, cat, bottle, truck, car, bird, dog]
    classes_str = ["knife", "keyboard", "elephant", "bicycle", "airplane", "clock",
                   "oven", "chair", "bear", "boat", "cat", "bottle", "truck", "car", "bird", "dog"]
    class_dict = dict(zip(classes_str, classes))
    reverse_dict = {}
    for class_name, imagenet_names in class_dict.items():
        for name in imagenet_names:
            reverse_dict[name] = class_name
    return class_dict, reverse_dict


def get_label_to_human_dict():
    names = """n02119789 1 kit_fox
n02100735 2 English_setter
n02110185 3 Siberian_husky
n02096294 4 Australian_terrier
n02102040 5 English_springer
n02066245 6 grey_whale
n02509815 7 lesser_panda
n02124075 8 Egyptian_cat
n02417914 9 ibex
n02123394 10 Persian_cat
n02125311 11 cougar
n02423022 12 gazelle
n02346627 13 porcupine
n02077923 14 sea_lion
n02110063 15 malamute
n02447366 16 badger
n02109047 17 Great_Dane
n02089867 18 Walker_hound
n02102177 19 Welsh_springer_spaniel
n02091134 20 whippet
n02092002 21 Scottish_deerhound
n02071294 22 killer_whale
n02442845 23 mink
n02504458 24 African_elephant
n02092339 25 Weimaraner
n02098105 26 soft-coated_wheaten_terrier
n02096437 27 Dandie_Dinmont
n02114712 28 red_wolf
n02105641 29 Old_English_sheepdog
n02128925 30 jaguar
n02091635 31 otterhound
n02088466 32 bloodhound
n02096051 33 Airedale
n02117135 34 hyena
n02138441 35 meerkat
n02097130 36 giant_schnauzer
n02493509 37 titi
n02457408 38 three-toed_sloth
n02389026 39 sorrel
n02443484 40 black-footed_ferret
n02110341 41 dalmatian
n02089078 42 black-and-tan_coonhound
n02086910 43 papillon
n02445715 44 skunk
n02093256 45 Staffordshire_bullterrier
n02113978 46 Mexican_hairless
n02106382 47 Bouvier_des_Flandres
n02441942 48 weasel
n02113712 49 miniature_poodle
n02113186 50 Cardigan
n02105162 51 malinois
n02415577 52 bighorn
n02356798 53 fox_squirrel
n02488702 54 colobus
n02123159 55 tiger_cat
n02098413 56 Lhasa
n02422699 57 impala
n02114855 58 coyote
n02094433 59 Yorkshire_terrier
n02111277 60 Newfoundland
n02132136 61 brown_bear
n02119022 62 red_fox
n02091467 63 Norwegian_elkhound
n02106550 64 Rottweiler
n02422106 65 hartebeest
n02091831 66 Saluki
n02120505 67 grey_fox
n02104365 68 schipperke
n02086079 69 Pekinese
n02112706 70 Brabancon_griffon
n02098286 71 West_Highland_white_terrier
n02095889 72 Sealyham_terrier
n02484975 73 guenon
n02137549 74 mongoose
n02500267 75 indri
n02129604 76 tiger
n02090721 77 Irish_wolfhound
n02396427 78 wild_boar
n02108000 79 EntleBucher
n02391049 80 zebra
n02412080 81 ram
n02108915 82 French_bulldog
n02480495 83 orangutan
n02110806 84 basenji
n02128385 85 leopard
n02107683 86 Bernese_mountain_dog
n02085936 87 Maltese_dog
n02094114 88 Norfolk_terrier
n02087046 89 toy_terrier
n02100583 90 vizsla
n02096177 91 cairn
n02494079 92 squirrel_monkey
n02105056 93 groenendael
n02101556 94 clumber
n02123597 95 Siamese_cat
n02481823 96 chimpanzee
n02105505 97 komondor
n02088094 98 Afghan_hound
n02085782 99 Japanese_spaniel
n02489166 100 proboscis_monkey
n02364673 101 guinea_pig
n02114548 102 white_wolf
n02134084 103 ice_bear
n02480855 104 gorilla
n02090622 105 borzoi
n02113624 106 toy_poodle
n02093859 107 Kerry_blue_terrier
n02403003 108 ox
n02097298 109 Scotch_terrier
n02108551 110 Tibetan_mastiff
n02493793 111 spider_monkey
n02107142 112 Doberman
n02096585 113 Boston_bull
n02107574 114 Greater_Swiss_Mountain_dog
n02107908 115 Appenzeller
n02086240 116 Shih-Tzu
n02102973 117 Irish_water_spaniel
n02112018 118 Pomeranian
n02093647 119 Bedlington_terrier
n02397096 120 warthog
n02437312 121 Arabian_camel
n02483708 122 siamang
n02097047 123 miniature_schnauzer
n02106030 124 collie
n02099601 125 golden_retriever
n02093991 126 Irish_terrier
n02110627 127 affenpinscher
n02106166 128 Border_collie
n02326432 129 hare
n02108089 130 boxer
n02097658 131 silky_terrier
n02088364 132 beagle
n02111129 133 Leonberg
n02100236 134 German_short-haired_pointer
n02486261 135 patas
n02115913 136 dhole
n02486410 137 baboon
n02487347 138 macaque
n02099849 139 Chesapeake_Bay_retriever
n02108422 140 bull_mastiff
n02104029 141 kuvasz
n02492035 142 capuchin
n02110958 143 pug
n02099429 144 curly-coated_retriever
n02094258 145 Norwich_terrier
n02099267 146 flat-coated_retriever
n02395406 147 hog
n02112350 148 keeshond
n02109961 149 Eskimo_dog
n02101388 150 Brittany_spaniel
n02113799 151 standard_poodle
n02095570 152 Lakeland_terrier
n02128757 153 snow_leopard
n02101006 154 Gordon_setter
n02115641 155 dingo
n02097209 156 standard_schnauzer
n02342885 157 hamster
n02097474 158 Tibetan_terrier
n02120079 159 Arctic_fox
n02095314 160 wire-haired_fox_terrier
n02088238 161 basset
n02408429 162 water_buffalo
n02133161 163 American_black_bear
n02328150 164 Angora
n02410509 165 bison
n02492660 166 howler_monkey
n02398521 167 hippopotamus
n02112137 168 chow
n02510455 169 giant_panda
n02093428 170 American_Staffordshire_terrier
n02105855 171 Shetland_sheepdog
n02111500 172 Great_Pyrenees
n02085620 173 Chihuahua
n02123045 174 tabby
n02490219 175 marmoset
n02099712 176 Labrador_retriever
n02109525 177 Saint_Bernard
n02454379 178 armadillo
n02111889 179 Samoyed
n02088632 180 bluetick
n02090379 181 redbone
n02443114 182 polecat
n02361337 183 marmot
n02105412 184 kelpie
n02483362 185 gibbon
n02437616 186 llama
n02107312 187 miniature_pinscher
n02325366 188 wood_rabbit
n02091032 189 Italian_greyhound
n02129165 190 lion
n02102318 191 cocker_spaniel
n02100877 192 Irish_setter
n02074367 193 dugong
n02504013 194 Indian_elephant
n02363005 195 beaver
n02102480 196 Sussex_spaniel
n02113023 197 Pembroke
n02086646 198 Blenheim_spaniel
n02497673 199 Madagascar_cat
n02087394 200 Rhodesian_ridgeback
n02127052 201 lynx
n02116738 202 African_hunting_dog
n02488291 203 langur
n02091244 204 Ibizan_hound
n02114367 205 timber_wolf
n02130308 206 cheetah
n02089973 207 English_foxhound
n02105251 208 briard
n02134418 209 sloth_bear
n02093754 210 Border_terrier
n02106662 211 German_shepherd
n02444819 212 otter
n01882714 213 koala
n01871265 214 tusker
n01872401 215 echidna
n01877812 216 wallaby
n01873310 217 platypus
n01883070 218 wombat
n04086273 219 revolver
n04507155 220 umbrella
n04147183 221 schooner
n04254680 222 soccer_ball
n02672831 223 accordion
n02219486 224 ant
n02317335 225 starfish
n01968897 226 chambered_nautilus
n03452741 227 grand_piano
n03642806 228 laptop
n07745940 229 strawberry
n02690373 230 airliner
n04552348 231 warplane
n02692877 232 airship
n02782093 233 balloon
n04266014 234 space_shuttle
n03344393 235 fireboat
n03447447 236 gondola
n04273569 237 speedboat
n03662601 238 lifeboat
n02951358 239 canoe
n04612504 240 yawl
n02981792 241 catamaran
n04483307 242 trimaran
n03095699 243 container_ship
n03673027 244 liner
n03947888 245 pirate
n02687172 246 aircraft_carrier
n04347754 247 submarine
n04606251 248 wreck
n03478589 249 half_track
n04389033 250 tank
n03773504 251 missile
n02860847 252 bobsled
n03218198 253 dogsled
n02835271 254 bicycle-built-for-two
n03792782 255 mountain_bike
n03393912 256 freight_car
n03895866 257 passenger_car
n02797295 258 barrow
n04204347 259 shopping_cart
n03791053 260 motor_scooter
n03384352 261 forklift
n03272562 262 electric_locomotive
n04310018 263 steam_locomotive
n02704792 264 amphibian
n02701002 265 ambulance
n02814533 266 beach_wagon
n02930766 267 cab
n03100240 268 convertible
n03594945 269 jeep
n03670208 270 limousine
n03770679 271 minivan
n03777568 272 Model_T
n04037443 273 racer
n04285008 274 sports_car
n03444034 275 go-kart
n03445924 276 golfcart
n03785016 277 moped
n04252225 278 snowplow
n03345487 279 fire_engine
n03417042 280 garbage_truck
n03930630 281 pickup
n04461696 282 tow_truck
n04467665 283 trailer_truck
n03796401 284 moving_van
n03977966 285 police_van
n04065272 286 recreational_vehicle
n04335435 287 streetcar
n04252077 288 snowmobile
n04465501 289 tractor
n03776460 290 mobile_home
n04482393 291 tricycle
n04509417 292 unicycle
n03538406 293 horse_cart
n03599486 294 jinrikisha
n03868242 295 oxcart
n02804414 296 bassinet
n03125729 297 cradle
n03131574 298 crib
n03388549 299 four-poster
n02870880 300 bookcase
n03018349 301 china_cabinet
n03742115 302 medicine_chest
n03016953 303 chiffonier
n04380533 304 table_lamp
n03337140 305 file
n03891251 306 park_bench
n02791124 307 barber_chair
n04429376 308 throne
n03376595 309 folding_chair
n04099969 310 rocking_chair
n04344873 311 studio_couch
n04447861 312 toilet_seat
n03179701 313 desk
n03982430 314 pool_table
n03201208 315 dining_table
n03290653 316 entertainment_center
n04550184 317 wardrobe
n07742313 318 Granny_Smith
n07747607 319 orange
n07749582 320 lemon
n07753113 321 fig
n07753275 322 pineapple
n07753592 323 banana
n07754684 324 jackfruit
n07760859 325 custard_apple
n07768694 326 pomegranate
n12267677 327 acorn
n12620546 328 hip
n13133613 329 ear
n11879895 330 rapeseed
n12144580 331 corn
n12768682 332 buckeye
n03854065 333 organ
n04515003 334 upright
n03017168 335 chime
n03249569 336 drum
n03447721 337 gong
n03720891 338 maraca
n03721384 339 marimba
n04311174 340 steel_drum
n02787622 341 banjo
n02992211 342 cello
n04536866 343 violin
n03495258 344 harp
n02676566 345 acoustic_guitar
n03272010 346 electric_guitar
n03110669 347 cornet
n03394916 348 French_horn
n04487394 349 trombone
n03494278 350 harmonica
n03840681 351 ocarina
n03884397 352 panpipe
n02804610 353 bassoon
n03838899 354 oboe
n04141076 355 sax
n03372029 356 flute
n11939491 357 daisy
n12057211 358 yellow_lady's_slipper
n09246464 359 cliff
n09468604 360 valley
n09193705 361 alp
n09472597 362 volcano
n09399592 363 promontory
n09421951 364 sandbar
n09256479 365 coral_reef
n09332890 366 lakeside
n09428293 367 seashore
n09288635 368 geyser
n03498962 369 hatchet
n03041632 370 cleaver
n03658185 371 letter_opener
n03954731 372 plane
n03995372 373 power_drill
n03649909 374 lawn_mower
n03481172 375 hammer
n03109150 376 corkscrew
n02951585 377 can_opener
n03970156 378 plunger
n04154565 379 screwdriver
n04208210 380 shovel
n03967562 381 plow
n03000684 382 chain_saw
n01514668 383 cock
n01514859 384 hen
n01518878 385 ostrich
n01530575 386 brambling
n01531178 387 goldfinch
n01532829 388 house_finch
n01534433 389 junco
n01537544 390 indigo_bunting
n01558993 391 robin
n01560419 392 bulbul
n01580077 393 jay
n01582220 394 magpie
n01592084 395 chickadee
n01601694 396 water_ouzel
n01608432 397 kite
n01614925 398 bald_eagle
n01616318 399 vulture
n01622779 400 great_grey_owl
n01795545 401 black_grouse
n01796340 402 ptarmigan
n01797886 403 ruffed_grouse
n01798484 404 prairie_chicken
n01806143 405 peacock
n01806567 406 quail
n01807496 407 partridge
n01817953 408 African_grey
n01818515 409 macaw
n01819313 410 sulphur-crested_cockatoo
n01820546 411 lorikeet
n01824575 412 coucal
n01828970 413 bee_eater
n01829413 414 hornbill
n01833805 415 hummingbird
n01843065 416 jacamar
n01843383 417 toucan
n01847000 418 drake
n01855032 419 red-breasted_merganser
n01855672 420 goose
n01860187 421 black_swan
n02002556 422 white_stork
n02002724 423 black_stork
n02006656 424 spoonbill
n02007558 425 flamingo
n02009912 426 American_egret
n02009229 427 little_blue_heron
n02011460 428 bittern
n02012849 429 crane
n02013706 430 limpkin
n02018207 431 American_coot
n02018795 432 bustard
n02025239 433 ruddy_turnstone
n02027492 434 red-backed_sandpiper
n02028035 435 redshank
n02033041 436 dowitcher
n02037110 437 oystercatcher
n02017213 438 European_gallinule
n02051845 439 pelican
n02056570 440 king_penguin
n02058221 441 albatross
n01484850 442 great_white_shark
n01491361 443 tiger_shark
n01494475 444 hammerhead
n01496331 445 electric_ray
n01498041 446 stingray
n02514041 447 barracouta
n02536864 448 coho
n01440764 449 tench
n01443537 450 goldfish
n02526121 451 eel
n02606052 452 rock_beauty
n02607072 453 anemone_fish
n02643566 454 lionfish
n02655020 455 puffer
n02640242 456 sturgeon
n02641379 457 gar
n01664065 458 loggerhead
n01665541 459 leatherback_turtle
n01667114 460 mud_turtle
n01667778 461 terrapin
n01669191 462 box_turtle
n01675722 463 banded_gecko
n01677366 464 common_iguana
n01682714 465 American_chameleon
n01685808 466 whiptail
n01687978 467 agama
n01688243 468 frilled_lizard
n01689811 469 alligator_lizard
n01692333 470 Gila_monster
n01693334 471 green_lizard
n01694178 472 African_chameleon
n01695060 473 Komodo_dragon
n01704323 474 triceratops
n01697457 475 African_crocodile
n01698640 476 American_alligator
n01728572 477 thunder_snake
n01728920 478 ringneck_snake
n01729322 479 hognose_snake
n01729977 480 green_snake
n01734418 481 king_snake
n01735189 482 garter_snake
n01737021 483 water_snake
n01739381 484 vine_snake
n01740131 485 night_snake
n01742172 486 boa_constrictor
n01744401 487 rock_python
n01748264 488 Indian_cobra
n01749939 489 green_mamba
n01751748 490 sea_snake
n01753488 491 horned_viper
n01755581 492 diamondback
n01756291 493 sidewinder
n01629819 494 European_fire_salamander
n01630670 495 common_newt
n01631663 496 eft
n01632458 497 spotted_salamander
n01632777 498 axolotl
n01641577 499 bullfrog
n01644373 500 tree_frog
n01644900 501 tailed_frog
n04579432 502 whistle
n04592741 503 wing
n03876231 504 paintbrush
n03483316 505 hand_blower
n03868863 506 oxygen_mask
n04251144 507 snorkel
n03691459 508 loudspeaker
n03759954 509 microphone
n04152593 510 screen
n03793489 511 mouse
n03271574 512 electric_fan
n03843555 513 oil_filter
n04332243 514 strainer
n04265275 515 space_heater
n04330267 516 stove
n03467068 517 guillotine
n02794156 518 barometer
n04118776 519 rule
n03841143 520 odometer
n04141975 521 scale
n02708093 522 analog_clock
n03196217 523 digital_clock
n04548280 524 wall_clock
n03544143 525 hourglass
n04355338 526 sundial
n03891332 527 parking_meter
n04328186 528 stopwatch
n03197337 529 digital_watch
n04317175 530 stethoscope
n04376876 531 syringe
n03706229 532 magnetic_compass
n02841315 533 binoculars
n04009552 534 projector
n04356056 535 sunglasses
n03692522 536 loupe
n04044716 537 radio_telescope
n02879718 538 bow
n02950826 539 cannon
n02749479 540 assault_rifle
n04090263 541 rifle
n04008634 542 projectile
n03085013 543 computer_keyboard
n04505470 544 typewriter_keyboard
n03126707 545 crane
n03666591 546 lighter
n02666196 547 abacus
n02977058 548 cash_machine
n04238763 549 slide_rule
n03180011 550 desktop_computer
n03485407 551 hand-held_computer
n03832673 552 notebook
n06359193 553 web_site
n03496892 554 harvester
n04428191 555 thresher
n04004767 556 printer
n04243546 557 slot
n04525305 558 vending_machine
n04179913 559 sewing_machine
n03602883 560 joystick
n04372370 561 switch
n03532672 562 hook
n02974003 563 car_wheel
n03874293 564 paddlewheel
n03944341 565 pinwheel
n03992509 566 potter's_wheel
n03425413 567 gas_pump
n02966193 568 carousel
n04371774 569 swing
n04067472 570 reel
n04040759 571 radiator
n04019541 572 puck
n03492542 573 hard_disc
n04355933 574 sunglass
n03929660 575 pick
n02965783 576 car_mirror
n04258138 577 solar_dish
n04074963 578 remote_control
n03208938 579 disk_brake
n02910353 580 buckle
n03476684 581 hair_slide
n03627232 582 knot
n03075370 583 combination_lock
n03874599 584 padlock
n03804744 585 nail
n04127249 586 safety_pin
n04153751 587 screw
n03803284 588 muzzle
n04162706 589 seat_belt
n04228054 590 ski
n02948072 591 candle
n03590841 592 jack-o'-lantern
n04286575 593 spotlight
n04456115 594 torch
n03814639 595 neck_brace
n03933933 596 pier
n04485082 597 tripod
n03733131 598 maypole
n03794056 599 mousetrap
n04275548 600 spider_web
n01768244 601 trilobite
n01770081 602 harvestman
n01770393 603 scorpion
n01773157 604 black_and_gold_garden_spider
n01773549 605 barn_spider
n01773797 606 garden_spider
n01774384 607 black_widow
n01774750 608 tarantula
n01775062 609 wolf_spider
n01776313 610 tick
n01784675 611 centipede
n01990800 612 isopod
n01978287 613 Dungeness_crab
n01978455 614 rock_crab
n01980166 615 fiddler_crab
n01981276 616 king_crab
n01983481 617 American_lobster
n01984695 618 spiny_lobster
n01985128 619 crayfish
n01986214 620 hermit_crab
n02165105 621 tiger_beetle
n02165456 622 ladybug
n02167151 623 ground_beetle
n02168699 624 long-horned_beetle
n02169497 625 leaf_beetle
n02172182 626 dung_beetle
n02174001 627 rhinoceros_beetle
n02177972 628 weevil
n02190166 629 fly
n02206856 630 bee
n02226429 631 grasshopper
n02229544 632 cricket
n02231487 633 walking_stick
n02233338 634 cockroach
n02236044 635 mantis
n02256656 636 cicada
n02259212 637 leafhopper
n02264363 638 lacewing
n02268443 639 dragonfly
n02268853 640 damselfly
n02276258 641 admiral
n02277742 642 ringlet
n02279972 643 monarch
n02280649 644 cabbage_butterfly
n02281406 645 sulphur_butterfly
n02281787 646 lycaenid
n01910747 647 jellyfish
n01914609 648 sea_anemone
n01917289 649 brain_coral
n01924916 650 flatworm
n01930112 651 nematode
n01943899 652 conch
n01944390 653 snail
n01945685 654 slug
n01950731 655 sea_slug
n01955084 656 chiton
n02319095 657 sea_urchin
n02321529 658 sea_cucumber
n03584829 659 iron
n03297495 660 espresso_maker
n03761084 661 microwave
n03259280 662 Dutch_oven
n04111531 663 rotisserie
n04442312 664 toaster
n04542943 665 waffle_iron
n04517823 666 vacuum
n03207941 667 dishwasher
n04070727 668 refrigerator
n04554684 669 washer
n03133878 670 Crock_Pot
n03400231 671 frying_pan
n04596742 672 wok
n02939185 673 caldron
n03063689 674 coffeepot
n04398044 675 teapot
n04270147 676 spatula
n02699494 677 altar
n04486054 678 triumphal_arch
n03899768 679 patio
n04311004 680 steel_arch_bridge
n04366367 681 suspension_bridge
n04532670 682 viaduct
n02793495 683 barn
n03457902 684 greenhouse
n03877845 685 palace
n03781244 686 monastery
n03661043 687 library
n02727426 688 apiary
n02859443 689 boathouse
n03028079 690 church
n03788195 691 mosque
n04346328 692 stupa
n03956157 693 planetarium
n04081281 694 restaurant
n03032252 695 cinema
n03529860 696 home_theater
n03697007 697 lumbermill
n03065424 698 coil
n03837869 699 obelisk
n04458633 700 totem_pole
n02980441 701 castle
n04005630 702 prison
n03461385 703 grocery_store
n02776631 704 bakery
n02791270 705 barbershop
n02871525 706 bookshop
n02927161 707 butcher_shop
n03089624 708 confectionery
n04200800 709 shoe_shop
n04443257 710 tobacco_shop
n04462240 711 toyshop
n03388043 712 fountain
n03042490 713 cliff_dwelling
n04613696 714 yurt
n03216828 715 dock
n02892201 716 brass
n03743016 717 megalith
n02788148 718 bannister
n02894605 719 breakwater
n03160309 720 dam
n03000134 721 chainlink_fence
n03930313 722 picket_fence
n04604644 723 worm_fence
n04326547 724 stone_wall
n03459775 725 grille
n04239074 726 sliding_door
n04501370 727 turnstile
n03792972 728 mountain_tent
n04149813 729 scoreboard
n03530642 730 honeycomb
n03961711 731 plate_rack
n03903868 732 pedestal
n02814860 733 beacon
n07711569 734 mashed_potato
n07720875 735 bell_pepper
n07714571 736 head_cabbage
n07714990 737 broccoli
n07715103 738 cauliflower
n07716358 739 zucchini
n07716906 740 spaghetti_squash
n07717410 741 acorn_squash
n07717556 742 butternut_squash
n07718472 743 cucumber
n07718747 744 artichoke
n07730033 745 cardoon
n07734744 746 mushroom
n04209239 747 shower_curtain
n03594734 748 jean
n02971356 749 carton
n03485794 750 handkerchief
n04133789 751 sandal
n02747177 752 ashcan
n04125021 753 safe
n07579787 754 plate
n03814906 755 necklace
n03134739 756 croquet_ball
n03404251 757 fur_coat
n04423845 758 thimble
n03877472 759 pajama
n04120489 760 running_shoe
n03062245 761 cocktail_shaker
n03014705 762 chest
n03717622 763 manhole_cover
n03777754 764 modem
n04493381 765 tub
n04476259 766 tray
n02777292 767 balance_beam
n07693725 768 bagel
n03998194 769 prayer_rug
n03617480 770 kimono
n07590611 771 hot_pot
n04579145 772 whiskey_jug
n03623198 773 knee_pad
n07248320 774 book_jacket
n04277352 775 spindle
n04229816 776 ski_mask
n02823428 777 beer_bottle
n03127747 778 crash_helmet
n02877765 779 bottlecap
n04435653 780 tile_roof
n03724870 781 mask
n03710637 782 maillot
n03920288 783 Petri_dish
n03379051 784 football_helmet
n02807133 785 bathing_cap
n04399382 786 teddy
n03527444 787 holster
n03983396 788 pop_bottle
n03924679 789 photocopier
n04532106 790 vestment
n06785654 791 crossword_puzzle
n03445777 792 golf_ball
n07613480 793 trifle
n04350905 794 suit
n04562935 795 water_tower
n03325584 796 feather_boa
n03045698 797 cloak
n07892512 798 red_wine
n03250847 799 drumstick
n04192698 800 shield
n03026506 801 Christmas_stocking
n03534580 802 hoopskirt
n07565083 803 menu
n04296562 804 stage
n02869837 805 bonnet
n07871810 806 meat_loaf
n02799071 807 baseball
n03314780 808 face_powder
n04141327 809 scabbard
n04357314 810 sunscreen
n02823750 811 beer_glass
n13052670 812 hen-of-the-woods
n07583066 813 guacamole
n03637318 814 lampshade
n04599235 815 wool
n07802026 816 hay
n02883205 817 bow_tie
n03709823 818 mailbag
n04560804 819 water_jug
n02909870 820 bucket
n03207743 821 dishrag
n04263257 822 soup_bowl
n07932039 823 eggnog
n03786901 824 mortar
n04479046 825 trench_coat
n03873416 826 paddle
n02999410 827 chain
n04367480 828 swab
n03775546 829 mixing_bowl
n07875152 830 potpie
n04591713 831 wine_bottle
n04201297 832 shoji
n02916936 833 bulletproof_vest
n03240683 834 drilling_platform
n02840245 835 binder
n02963159 836 cardigan
n04370456 837 sweatshirt
n03991062 838 pot
n02843684 839 birdhouse
n03482405 840 hamper
n03942813 841 ping-pong_ball
n03908618 842 pencil_box
n03902125 843 pay-phone
n07584110 844 consomme
n02730930 845 apron
n04023962 846 punching_bag
n02769748 847 backpack
n10148035 848 groom
n02817516 849 bearskin
n03908714 850 pencil_sharpener
n02906734 851 broom
n03788365 852 mosquito_net
n02667093 853 abaya
n03787032 854 mortarboard
n03980874 855 poncho
n03141823 856 crutch
n03976467 857 Polaroid_camera
n04264628 858 space_bar
n07930864 859 cup
n04039381 860 racket
n06874185 861 traffic_light
n04033901 862 quill
n04041544 863 radio
n07860988 864 dough
n03146219 865 cuirass
n03763968 866 military_uniform
n03676483 867 lipstick
n04209133 868 shower_cap
n03782006 869 monitor
n03857828 870 oscilloscope
n03775071 871 mitten
n02892767 872 brassiere
n07684084 873 French_loaf
n04522168 874 vase
n03764736 875 milk_can
n04118538 876 rugby_ball
n03887697 877 paper_towel
n13044778 878 earthstar
n03291819 879 envelope
n03770439 880 miniskirt
n03124170 881 cowboy_hat
n04487081 882 trolleybus
n03916031 883 perfume
n02808440 884 bathtub
n07697537 885 hotdog
n12985857 886 coral_fungus
n02917067 887 bullet_train
n03938244 888 pillow
n15075141 889 toilet_tissue
n02978881 890 cassette
n02966687 891 carpenter's_kit
n03633091 892 ladle
n13040303 893 stinkhorn
n03690938 894 lotion
n03476991 895 hair_spray
n02669723 896 academic_gown
n03220513 897 dome
n03127925 898 crate
n04584207 899 wig
n07880968 900 burrito
n03937543 901 pill_bottle
n03000247 902 chain_mail
n04418357 903 theater_curtain
n04590129 904 window_shade
n02795169 905 barrel
n04553703 906 washbasin
n02783161 907 ballpoint
n02802426 908 basketball
n02808304 909 bath_towel
n03124043 910 cowboy_boot
n03450230 911 gown
n04589890 912 window_screen
n12998815 913 agaric
n02992529 914 cellular_telephone
n03825788 915 nipple
n02790996 916 barbell
n03710193 917 mailbox
n03630383 918 lab_coat
n03347037 919 fire_screen
n03769881 920 minibus
n03871628 921 packet
n03733281 922 maze
n03976657 923 pole
n03535780 924 horizontal_bar
n04259630 925 sombrero
n03929855 926 pickelhaube
n04049303 927 rain_barrel
n04548362 928 wallet
n02979186 929 cassette_player
n06596364 930 comic_book
n03935335 931 piggy_bank
n06794110 932 street_sign
n02825657 933 bell_cote
n03388183 934 fountain_pen
n04591157 935 Windsor_tie
n04540053 936 volleyball
n03866082 937 overskirt
n04136333 938 sarong
n04026417 939 purse
n02865351 940 bolo_tie
n02834397 941 bib
n03888257 942 parachute
n04235860 943 sleeping_bag
n04404412 944 television
n04371430 945 swimming_trunks
n03733805 946 measuring_cup
n07920052 947 espresso
n07873807 948 pizza
n02895154 949 breastplate
n04204238 950 shopping_basket
n04597913 951 wooden_spoon
n04131690 952 saltshaker
n07836838 953 chocolate_sauce
n09835506 954 ballplayer
n03443371 955 goblet
n13037406 956 gyromitra
n04336792 957 stretcher
n04557648 958 water_bottle
n03187595 959 dial_telephone
n04254120 960 soap_dispenser
n03595614 961 jersey
n04146614 962 school_bus
n03598930 963 jigsaw_puzzle
n03958227 964 plastic_bag
n04069434 965 reflex_camera
n03188531 966 diaper
n02786058 967 Band_Aid
n07615774 968 ice_lolly
n04525038 969 velvet
n04409515 970 tennis_ball
n03424325 971 gasmask
n03223299 972 doormat
n03680355 973 Loafer
n07614500 974 ice_cream
n07695742 975 pretzel
n04033995 976 quilt
n03710721 977 maillot
n04392985 978 tape_player
n03047690 979 clog
n03584254 980 iPod
n13054560 981 bolete
n10565667 982 scuba_diver
n03950228 983 pitcher
n03729826 984 matchstick
n02837789 985 bikini
n04254777 986 sock
n02988304 987 CD_player
n03657121 988 lens_cap
n04417672 989 thatch
n04523525 990 vault
n02815834 991 beaker
n09229709 992 bubble
n07697313 993 cheeseburger
n03888605 994 parallel_bars
n03355925 995 flagpole
n03063599 996 coffee_mug
n04116512 997 rubber_eraser
n04325704 998 stole
n07831146 999 carbonara
n03255030 1000 dumbbell"""
    lines = names.split("\n")
    labels_to_human = {}
    for line in lines:
        class_label, _, human_name = line.split()
        labels_to_human[class_label] = human_name
    return labels_to_human


COCO_IMAGENET_CLASS_DICT, COCO_IMAGENET_REVERSE_DICT = get_coco_dicts()
LABEL_TO_HUMAN_DICT = get_label_to_human_dict()
