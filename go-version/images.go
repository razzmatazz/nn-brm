package main

import "regexp"
import "strconv"
import "errors"
import "fmt"
import "os"
import "bufio"

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

