package main

import "fmt"
import "math"
import "math/rand"
import "errors"
import "os"
import "log"
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

	for _, i := range inputList {
		input := i // bind it locally for closure to work
		go func () {
			gradientChan <- net.trainInput(input)
		}()
	}

	// perform a reduce
	delta := make([]connWeight, len(net.Weights))

	for i := 0; i < len(inputList); i++ {
		gradient := <- gradientChan

		//log.Printf("applying %v, sum=%v", i, deltaSum(gradient))
		delta = gradientSum(delta, gradient)
	}

	//log.Printf("---applying=%v", deltaSum(delta))
	//panic("done")
	net.applyWeightDeltas(delta)
	
	return nil
}

func deltaSum(deltas []connWeight) float64 {
	sum := float64(.0)

	for _, v := range deltas {
		sum += math.Abs(float64(v))
	}

	return sum
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

	return enc.Encode(net)
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
