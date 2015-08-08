package main

import "fmt"
import "math"
import "math/rand"
import "errors"
import "os"
import "bufio"
import "log"
import "regexp"
import "strconv"
import "time"
import "encoding/json"

type connWeight float64

func stochastic(w connWeight) float64 {
	return 1 / (1 + math.Exp(-float64(w)))
}

func sto(w connWeight) bool {
	if rand.Float64() < stochastic(w) {
		return true
	} else {
		return false
	}
}

type Network struct {
	Filename string
	Insz int
	Hiddensz int
	Weights []connWeight
}

func makeRandomWeights(size int) []connWeight {
	Weights := make([]connWeight, size, size)

	for i := 0; i < size; i++ {
		Weights[i] = connWeight(rand.Float64() - 0.5) * 0.1
	}

	return Weights
}

func NewNetwork(filename string, insz int, hiddensz int) *Network {

	net := &Network{
		Filename: filename,
		Insz: insz,
		Hiddensz: hiddensz,

		Weights: makeRandomWeights(insz * hiddensz) }

	return net
}

func (net *Network) TrainInputList(inputList [][]bool) error {
	// check some things about 'inputList'
	
	if len(inputList) == 0 {
		return errors.New("Network.TrainInputList: input list is empty")
	}

	for i := 0; i < len(inputList); i++ {
		if len(inputList[i]) != net.Insz {
			return errors.New(fmt.Sprintf(
				"Network.TrainInputList: input #%v has size %v, but network input size is %v",
				i, len(inputList[i]), net.Insz))
		}
	}

	// train the network on a set of input: first do a map for all inputs
	gradientChan := make(chan []connWeight)

	for _, input := range inputList {
		go func () { gradientChan <- net.trainInput(input) }()
	}

	// perform a reduce
	for i := 0; i < len(inputList); i++ {
		net.applyWeightDeltas(<- gradientChan)
	}
	
	return nil
}

func (net* Network) computeGradient(input []bool, output []bool) []connWeight {

	gradient := make([]connWeight, net.Insz * net.Hiddensz)

	for h := 0; h < net.Hiddensz; h++ {
		for i := 0; i < net.Insz; i++ {
			var g connWeight

			if input[i] && output[h] {
				g = 1
			} else {
				g = 0
			}

			gradient[i * net.Hiddensz + h] = g
		}
	}

	return gradient
}

func gradientMul(factor connWeight, gradients []connWeight) []connWeight {
	mul := make([]connWeight, len(gradients))

	for i := 0; i < len(gradients); i++ {
		mul[i] = gradients[i] * factor
	}

	return mul
}

func gradientSum(gradientsA []connWeight, gradientsB []connWeight) []connWeight {

	sum := make([]connWeight, len(gradientsA))

	for i := 0; i < len(gradientsA); i++ {
		sum[i] = gradientsA[i] + gradientsB[i]
	}

	return sum
}

func (net *Network) trainInput(input []bool) []connWeight {
	if len(input) != net.Insz {
		panic("Network.trainInput: input len != net.Insz")
	}

	hidden := net.Forward(input)

	pos_gradient := net.computeGradient(input, hidden)

	input_mod := net.Backwards(hidden)
	hidden_mod := net.Forward(input_mod)

	neg_gradient := net.computeGradient(input_mod, hidden_mod)
	neg_gradient = gradientMul(-1.0, neg_gradient)

	return gradientMul(0.04, gradientSum(pos_gradient, neg_gradient))
}

func (net *Network) Forward(input []bool) []bool {
	if len(input) != net.Insz {
		log.Panic(errors.New(fmt.Sprintf(
			"Network.Forward: input size (%v) != net.Insz (%v)",
			len(input),
			net.Insz)))
	}
	
	output := make([]bool, net.Hiddensz)

	for o := 0; o < net.Hiddensz; o++ {
		sum := connWeight(.0)

		for i, in := range input {
			if in {
				sum += net.Weights[i * net.Hiddensz + o]
			}
		}

		output[o] = sto(sum)
	}
		
	return output
}

func (net *Network) Backwards(output []bool) []bool {
	input := make([]bool, net.Insz)

	for i := 0; i < net.Insz; i++ {
		sum := connWeight(.0)

		for o, out := range output {
			if out {
				sum += net.Weights[i * net.Hiddensz + o]
			}
		}

		input[i] = sto(sum)
	}

	return input
}

func (net *Network) applyWeightDeltas(deltas []connWeight) {
	for i, d := range deltas {
		net.Weights[i] += d
	}
}

func (net *Network) reflect(input []bool) []bool {
	output := net.Forward(input)
	return net.Backwards(output)
}

func (net *Network) Save() error {
	file, err := os.Create(net.Filename)
	if err != nil {
		return err
	}

	defer file.Close()

	enc := json.NewEncoder(file)

	enc.Encode(net)

	return nil
}

func LoadNetwork(filename string) (*Network, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	net := &Network{}

	err = json.NewDecoder(file).Decode(net)
	if err != nil {
		return nil, err
	}

	return net, nil
}

func parseFormatString(fmt string) (int, int, error) {
	re := regexp.MustCompile("^(\\d+)\\s*x\\s*(\\d+)$")
	
	results := re.FindStringSubmatch(fmt)

	x, err := strconv.ParseUint(results[1], 10, 32)
	if err != nil {
		return 0, 0, err
	}

	y, err := strconv.ParseUint(results[2], 10, 32)
	if err != nil {
		return 0, 0, err
	}

	return int(x), int(y), nil
}

func parseImageLine(line string, imgWidth int) ([]bool, error) {
	if len(line) != imgWidth + 2 {
		return nil, errors.New("invalid line width")
	}

	if line[0] != '"' || line[imgWidth + 1] != '"' {
		return nil, errors.New(fmt.Sprintf("image line \"%s\" does not start and end with \"", line))
	}

	dataRunes := line[1:imgWidth + 1]

	pixels := make([]bool, 0, imgWidth)

	for _, rune := range dataRunes {
		if rune == '#' || rune == '.' {
			pixels = append(pixels, true)
		} else if rune == ' ' {
			pixels = append(pixels, false)
		} else {
			return nil, errors.New(fmt.Sprintf("invalid image character \"%c\"", rune))
		}
	}

	return pixels, nil
}

func loadImages(filename string) ([][]bool, int, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, 0, err
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)

	scanner.Scan()
	
	imgWidth, imgHeight, err := parseFormatString(scanner.Text())

	if err != nil {
		return nil, 0, 0, err
	}

	images := make([][]bool, 0, 32)

	var image []bool

	for scanner.Scan() {
		line := scanner.Text()

		if len(line) != 0 {
			if image == nil {
				image = make([]bool, 0, imgWidth * imgHeight)
			}

			pixels, err := parseImageLine(line, imgWidth)
			if err != nil {
				return nil, 0, 0, err
			}

			for _, pix := range pixels {
				image = append(image, pix)
			}

			if len(image) == imgWidth * imgHeight {
				images = append(images, image)
				image = nil
			}
		}
	}

	if image != nil {
		return nil, 0, 0, errors.New("unfinished image at the end of the file")
	}

	return images, imgWidth, imgHeight, nil
}

func printImages(images [][]bool, imgWidth int, imgHeight int) error {

	for _, image := range images {
		if len(image) != imgWidth * imgHeight {
			return errors.New("unexpected image bool[] size")
		}
	}

	for i := 0; i < imgHeight; i++ {
		for _, image := range images {
			for _, pix := range image[i * imgHeight: i * imgHeight + imgWidth] {
				if pix {
					fmt.Printf("#")
				} else {
					fmt.Printf(" ")
				}
			}

			fmt.Printf(" ")
		}

		fmt.Printf("\n")
	}

	// print footer line
	for i := 0; i < len(images); i++ {
		for p := 0; p < imgWidth; p++ {
			fmt.Printf("-")
		}

		fmt.Printf(" ")
	}

	fmt.Printf("\n")

	return nil
}

func main () {
	imagesFile := "./learning-images.txt"

	images, imgWidth, imgHeight, err := loadImages(imagesFile)
	if err != nil {
		log.Panic(err)
	}

	layer1Filename := "_layer1.json"

	net, err := LoadNetwork(layer1Filename)
	if err != nil {
		log.Printf("cannot load network from file \"%v\": %v; will make new network from scratch", layer1Filename, err)

		net = NewNetwork(layer1Filename, 5 * 5, 32)
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
		net.Save()
	}
}
