IMAGES_EMPTY = [
     
    "     "
    "     "
    "     "
    "     "
    "     "
]


IMAGES_1 = [
     
    "     "
    "  #  "
    "  #  "
    "  #  "
    "  #  ",

    "     "
    " #   "
    " #   "
    " #   "
    ".##  ",


    "     "
    " #   "
    " #   "
    " #   "
    " #   ",

    "     "
    "  #  "
    "  #  "
    " #   "
    " #   ",

    "     "
    "#    "
    "#    "
    " #   "
    " #   ",

    "     "
    "     "
    "#    "
    "#    "
    "#    ",

    "     "
    "   # "
    "   # "
    "   # "
    "   # ",

    "     "
    "     "
    " #   "
    " #   "
    " #   ",

    "     "
    "#    "
    " #   "
    " #   "
    " #.  ",

    "     "
    "#    "
    " #   "
    " #   "
    " #   ",

    "     "
    "  #  "
    " #   "
    " #   "
    "  #  ",

    "     "
    "  #  "
    "   # "
    "   # "
    "   # ",

    "     "
    "#.   "
    "#    "
    "#    "
    ".#   ",
    
    "     "
    "   # "
    "  .# "
    "  #  "
    "  #  ",
]

IMAGES_2 = [
     
    "     "
    "###  "
    "  .# "
    ".##  "
    "#### ",
    
    "     "
    ".##  "
    "  .# "
    ".#.  "
    ".##. ",
    
    "     "
    "  ## "
    "  .# "
    " #.  "
    " ### ",

    "     "
    " ##  "
    " .#  "
    "#.   "
    "###  "
]

IMAGES_7 = [
     
    "     "
    " ### "
    "   # "
    "  #  "
    "  #  ",
    
    "     "
    "  ## "
    "  .# "
    ".#   "
    ".#   ",
    
    "     "
    "#### "
    "  .# "
    "###  "
    " #   ",

    "     "
    "###  "
    " .#  "
    " #   "
    "#    ",
    
    "     "
    "     "
    " ### "
    "  #  "
    " #   ",
    
    "     "
    " ### "
    "  #  "
    " #   "
    "     "
]

IMAGES_A = [
     
    " ### "
    "#  # "
    "#   #"
    "### #"
    "#  # ",
    
    "  #  "
    " ### "
    " # # "
    " ####"
    " #  #",
    
    "   # "
    " ### "
    "#  # "
    "#### "
    "#   #",

    "     "
    "###  "
    "# #  "
    "###  "
    "# #  ",
    
    "   ##"
    " ## #"
    " #  #"
    "  ###"
    " #  #",
    
    "     "
    " ### "
    "#   #"
    " ### "
    "#   #"
]


IMAGES = IMAGES_EMPTY + IMAGES_1 + IMAGES_2 + IMAGES_7 + IMAGES_A
IMAGE_DIM = 5

def num_to_char(n):
    return (" " if n < .2
            else ("." if n < 0.4
                  else ("+" if n < 0.6
                        else ("x" if n < 0.8
                              else "#"))))

def print_images(imgs):
    print ("-" * IMAGE_DIM + " ") * len(imgs)
    for line in range(IMAGE_DIM):
        for img in imgs:
            print "".join(map(num_to_char, img[line * IMAGE_DIM:][:IMAGE_DIM])),

        print ""
            

def read_image(ascii):
    for a in ascii:
        if a == " ":
            yield .0
        elif a == ".":
            yield .3
        elif a == "#":
            yield 1
        else:
            raise Exception("invalid image char '" + a + "'")

