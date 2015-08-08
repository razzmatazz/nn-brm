_dummy_:
	go build main.go net.go images.go
	go build dream.go net.go images.go

train:
	GOMAXPROCS=8 go run main.go net.go images.go

retrain:
	rm _layer1.json || true
	GOMAXPROCS=8 go run main.go net.go images.go

dream:
	go run dream.go
