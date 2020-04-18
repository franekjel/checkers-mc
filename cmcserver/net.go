package main

import (
	"encoding/binary"
	"log"
	"net"
	"strconv"
	"time"
)

type connectionData struct {
	conn net.Conn
	data []byte
}

//reads 4 bytes and convert it to uint
func readUint(conn net.Conn) uint {
	buff := readNBytes(conn, 4)
	if buff != nil {
		return uint(binary.BigEndian.Uint32(buff))
	}
	return 0
}

//reads n bytes from conn
func readNBytes(conn net.Conn, n uint) []byte {
	buff := make([]byte, n)
	var i uint
	for i < n {
		nbyte, err := conn.Read(buff[i:n])
		i += uint(nbyte)
		if err != nil {
			log.Printf("Cannot read message: %s", err.Error())
			return nil
		}
	}
	return buff
}

//send given uint
func sendUint(conn net.Conn, n uint32) {
	buff := make([]byte, 4)
	binary.BigEndian.PutUint32(buff, n)
	var i int
	for i < 4 {
		n, err := conn.Write(buff[i:4])
		i += n
		if err != nil {
			log.Printf("Cannot send data to client %s. %s", conn.RemoteAddr().String(), err.Error())
			return
		}
	}
}

//send given slice
func sendSlice(conn net.Conn, buff []byte) {
	var i = 0
	n := len(buff)
	for i < n {
		nbyte, err := conn.Write(buff[i:n])
		i += nbyte
		if err != nil {
			log.Printf("Cannot send slice to client")
			return
		}
	}
}

//reads data from client and send to chan
func readMessage(conn net.Conn, ch chan *connectionData) {
	conn.SetDeadline(time.Now().Add(3 * time.Second))
	msgSize := readUint(conn)
	if msgSize == 0 || msgSize > 128*1024 {
		log.Printf("Bad message size or other error from %s", conn.RemoteAddr().String())
		conn.Close()
		return
	}
	buff := readNBytes(conn, msgSize)
	if buff == nil {
		log.Printf("Cannot get data from %s", conn.RemoteAddr().String())
		conn.Close()
		return
	}
	data := connectionData{
		conn,
		buff,
	}
	ch <- &data
}

func startListening(port uint16, ch chan *connectionData) {
	l, err := net.Listen("tcp", ":"+strconv.FormatUint(uint64(port), 10))
	if err != nil {
		log.Fatalf("Cannot start listening on port %s: %s", strconv.FormatUint(uint64(port), 10), err.Error())
	}
	defer l.Close()
	log.Printf("Starting listening on port %d", port)

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Printf("Cannot accept connection: %s", err.Error())
			continue
		}
		log.Print("Got connection from ", conn.RemoteAddr())
		go readMessage(conn, ch)
	}
}

func sendResponse(conn net.Conn, buff []byte) {
	sendUint(conn, uint32(len(buff)))
	sendSlice(conn, buff)
	conn.Close()
}
