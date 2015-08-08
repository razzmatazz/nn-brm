package main

import "time"
import "log"
import "math/rand"


func main () {
	imagesFile := "./learning-images.txt"

	images, imgWidth, imgHeight, err := loadImages(imagesFile)
	if err != nil {
		log.Panic(err)
	}

	layer1Filename := "_layer1.json"

	net, err := LoadNetwork(layer1Filename)
	if err != nil {
		log.Printf(
			"cannot load network from file \"%v\": %v; will make new network from scratch",
			layer1Filename, err)

		net = NewNetwork(layer1Filename, 5 * 5, 64)
	}
	
	printImages(images[:4], imgWidth, imgHeight)

	printedLastTime := time.Now()

	for ;; {
		err = net.TrainInputList(images)
		if err != nil {
			log.Panic(err)
		}

		// show how input changes
		if time.Now().Unix() - printedLastTime.Unix() != 0 {

			imgIndex := rand.Intn(len(images))
			
			printImages(
				[][]bool { images[imgIndex], net.reflect(images[imgIndex]) },
				imgWidth,
				imgHeight)

			printedLastTime = time.Now()
		}

		// write network state to disk
		if err := net.Save(); err != nil {
			panic(err)
		}
		
	}
}
