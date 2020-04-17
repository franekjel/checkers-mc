package main

import (
	"log"
	"os"
	"os/exec"
	"strings"

	flag "github.com/spf13/pflag"
)

func main() {
	checkersPath := flag.StringP("executable", "e", "./checkers-mc", "path to checkers-mc executable")
	port := flag.Uint16P("port", "p", 11821, "port to listening")
	flag.Parse()
	checkExecutable(*checkersPath)
	ch := make(chan *connectionData)
	startListening(*port, ch)
}

func checkExecutable(path string) {
	cmd := exec.Command(path, "-h")
	out, err := cmd.CombinedOutput()
	if err != nil && err.(*exec.ExitError).ExitCode() != 1 {
		log.Fatal("Error launching ", path, ": ", err.Error())
	}
	if !strings.Contains(string(out), "find best move in checkers") {
		log.Print(path, " -h  output is:\n", string(out))
		log.Fatal("Program ", path, " is not checkers-mc program")
		os.Exit(1)
	}
	log.Print("Program ", path, " seems to be checkers-mc")
}
