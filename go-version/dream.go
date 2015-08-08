package main

import "log"
import "time"
import "math/rand"

func main() {
	rand.Seed(time.Now().Unix())
	
	layer1Filename := "_layer1.json"

	net, err := LoadNetwork(layer1Filename)
	if err != nil {
		log.Panic(err)
	}

	imagesFile := "./learning-images.txt"
	_, imgWidth, imgHeight, err := loadImages(imagesFile)
	if err != nil {
		log.Panic(err)
	}

	input := make([]bool, imgWidth * imgHeight)

	for i := 0; i < imgWidth * imgHeight; i++ {
		if rand.Intn(8) == 0 {
			input[i] = true
		}
	}

	for ;; {
		input := net.reflect(input)

		printImages([][]bool { input }, imgWidth, imgHeight)

		time.Sleep(time.Millisecond * 125)
	}
}
