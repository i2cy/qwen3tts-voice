package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/mewkiz/flac"
)

type Config struct {
	Listen           string          `json:"listen"`
	Backend          string          `json:"backend"`
	BackendTimeoutMs int             `json:"backendTimeoutMs"`
	APIKeys          []string        `json:"apiKeys"`
	RequireAuth      bool            `json:"requireAuth"`
	Defaults         Defaults        `json:"defaults"`
	Cache            CacheConfig     `json:"cache"`
	Health           HealthConfig    `json:"health"`
}

type Defaults struct {
	Voice         string `json:"voice"`
	Language      string `json:"language"`
	MaxTokens     int    `json:"maxTokens"`
	Format        string `json:"format"`
	BackendFormat string `json:"backendFormat"`
}

type CacheConfig struct {
	MaxEntries int `json:"maxEntries"`
	TTLSeconds int `json:"ttlSeconds"`
}

type HealthConfig struct {
	BackendPath string `json:"backendPath"`
}

var cfg Config

// ---- LRU cache ----
type cacheEntry struct {
	data       []byte
	contentType string
	expiresAt  time.Time
}

type LRU struct {
	mu    sync.Mutex
	items map[string]*listNode
	list  *list
	max   int
	ttl   time.Duration
}

type listNode struct {
	key   string
	value cacheEntry
	prev  *listNode
	next  *listNode
}

type list struct {
	head *listNode
	tail *listNode
	size int
}

func newLRU(max int, ttl time.Duration) *LRU {
	return &LRU{items: make(map[string]*listNode, max), max: max, ttl: ttl, list: &list{}}
}

func (l *LRU) get(key string) (cacheEntry, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	n, ok := l.items[key]
	if !ok {
		return cacheEntry{}, false
	}
	if time.Now().After(n.value.expiresAt) {
		l.remove(n)
		return cacheEntry{}, false
	}
	l.moveToFront(n)
	return n.value, true
}

func (l *LRU) put(key string, v cacheEntry) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if n, ok := l.items[key]; ok {
		n.value = v
		l.moveToFront(n)
		return
	}
	n := &listNode{key: key, value: v}
	l.items[key] = n
	l.pushFront(n)
	if l.list.size > l.max {
		l.remove(l.list.tail)
	}
}

func (l *LRU) moveToFront(n *listNode) {
	if l.list.head == n {
		return
	}
	l.unlink(n)
	l.pushFront(n)
}

func (l *LRU) unlink(n *listNode) {
	if n.prev != nil {
		n.prev.next = n.next
	}
	if n.next != nil {
		n.next.prev = n.prev
	}
	if l.list.head == n {
		l.list.head = n.next
	}
	if l.list.tail == n {
		l.list.tail = n.prev
	}
	l.list.size--
}

func (l *LRU) pushFront(n *listNode) {
	n.prev = nil
	n.next = l.list.head
	if l.list.head != nil {
		l.list.head.prev = n
	}
	l.list.head = n
	if l.list.tail == nil {
		l.list.tail = n
	}
	l.list.size++
}

func (l *LRU) remove(n *listNode) {
	l.unlink(n)
	delete(l.items, n.key)
}

// ---- FLAC -> WAV ----
func flacToWav(data []byte) ([]byte, error) {
	stream, err := flac.Parse(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("flac parse: %w", err)
	}
	defer stream.Close()
	sr := int(stream.Info.SampleRate)
	ch := int(stream.Info.NChannels)
	bps := int(stream.Info.BitsPerSample)
	if bps != 16 {
		return nil, fmt.Errorf("unsupported bps %d", bps)
	}

	var pcm bytes.Buffer
	for {
		f, err := stream.ParseNext()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("flac next: %w", err)
		}
		for _, sub := range f.Subframes {
			// samples are s32; write interleaved s16 little-endian
			for _, s := range sub.Samples {
				v := int16(s)
				pcm.WriteByte(byte(v))
				pcm.WriteByte(byte(v >> 8))
			}
		}
	}

	// build WAV (RIFF) header + PCM
	dataSize := pcm.Len()
	var out bytes.Buffer
	out.WriteString("RIFF")
	binary.Write(&out, binary.LittleEndian, uint32(36+dataSize))
	out.WriteString("WAVE")
	out.WriteString("fmt ")
	binary.Write(&out, binary.LittleEndian, uint32(16))
	binary.Write(&out, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(&out, binary.LittleEndian, uint16(ch))
	binary.Write(&out, binary.LittleEndian, uint32(sr))
	binary.Write(&out, binary.LittleEndian, uint32(sr*ch*2))
	binary.Write(&out, binary.LittleEndian, uint16(ch*2))
	binary.Write(&out, binary.LittleEndian, uint16(16))
	out.WriteString("data")
	binary.Write(&out, binary.LittleEndian, uint32(dataSize))
	out.Write(pcm.Bytes())
	return out.Bytes(), nil
}

// ---- handlers ----
var cache *LRU
var httpClient *http.Client

func authOK(r *http.Request) bool {
	if !cfg.RequireAuth {
		return true
	}
	h := r.Header.Get("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		tok := strings.TrimPrefix(h, "Bearer ")
		for _, k := range cfg.APIKeys {
			if k == tok {
				return true
			}
		}
	}
	key := r.Header.Get("X-API-Key")
	if key != "" {
		for _, k := range cfg.APIKeys {
			if k == key {
				return true
			}
		}
	}
	return false
}

func cacheKey(req map[string]interface{}) string {
	b, _ := json.Marshal(req)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func handleSpeech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !authOK(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
		return
	}

	// merge defaults
	def := cfg.Defaults
	if _, ok := req["voice"]; !ok {
		req["voice"] = def.Voice
	}
	if _, ok := req["language"]; !ok {
		req["language"] = def.Language
	}
	if _, ok := req["max_tokens"]; !ok {
		req["max_tokens"] = def.MaxTokens
	}
	if _, ok := req["format"]; !ok {
		req["format"] = def.Format
	}
	if _, ok := req["backendFormat"]; !ok {
		req["backendFormat"] = def.BackendFormat
	}

	text, _ := req["input"].(string)
	if strings.TrimSpace(text) == "" {
		http.Error(w, "input required", http.StatusBadRequest)
		return
	}

	// cache lookup (key = request WITHOUT format, so wav+flac share one entry)
	lookup := make(map[string]interface{})
	for k, v := range req {
		if k == "format" || k == "backendFormat" {
			continue
		}
		lookup[k] = v
	}
	ck := cacheKey(lookup)
	if ent, ok := cache.get(ck); ok {
		w.Header().Set("Content-Type", ent.contentType)
		w.Header().Set("X-Cache", "HIT")
		w.Header().Set("Content-Length", fmt.Sprint(len(ent.data)))
		w.WriteHeader(http.StatusOK)
		w.Write(ent.data)
		return
	}

	// forward to backend with backendFormat (compressed transport)
	backendReq := make(map[string]interface{})
	for k, v := range req {
		if k == "format" {
			continue
		}
		backendReq[k] = v
	}
	backendReq["format"] = req["backendFormat"]

	outReq, err := json.Marshal(backendReq)
	if err != nil {
		http.Error(w, "marshal: "+err.Error(), http.StatusInternalServerError)
		return
	}
	bresp, err := httpClient.Post(cfg.Backend+"/v1/audio/speech", "application/json", bytes.NewReader(outReq))
	if err != nil {
		http.Error(w, "backend: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer bresp.Body.Close()
	data, err := io.ReadAll(bresp.Body)
	if err != nil {
		http.Error(w, "backend read: "+err.Error(), http.StatusBadGateway)
		return
	}
	if bresp.StatusCode != http.StatusOK {
		http.Error(w, "backend error "+bresp.Status+": "+string(data), bresp.StatusCode)
		return
	}

	outFmt, _ := req["format"].(string)
	outFmt = strings.ToLower(outFmt)
	contentType := "audio/wav"
	if outFmt == "flac" {
		contentType = "audio/flac"
	} else if outFmt == "wav" && strings.ToLower(req["backendFormat"].(string)) == "flac" {
		data, err = flacToWav(data)
		if err != nil {
			http.Error(w, "flac->wav: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	ent := cacheEntry{data: data, contentType: contentType, expiresAt: time.Now().Add(cache.ttl)}
	cache.put(ck, ent)

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("X-Cache", "MISS")
	w.Header().Set("Content-Length", fmt.Sprint(len(data)))
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	info := map[string]interface{}{
		"ok":       true,
		"service":  "qwen3tts-gateway",
		"backend":  cfg.Backend,
		"cache":    map[string]int{"maxEntries": cfg.Cache.MaxEntries, "ttlSeconds": cfg.Cache.TTLSeconds},
		"auth":     cfg.RequireAuth,
	}
	if bresp, err := httpClient.Get(cfg.Backend + cfg.Health.BackendPath); err == nil {
		defer bresp.Body.Close()
		b, _ := io.ReadAll(bresp.Body)
		info["backend_ok"] = bresp.StatusCode == http.StatusOK
		info["backend_body"] = string(b)
	} else {
		info["backend_ok"] = false
		info["backend_err"] = err.Error()
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

func main() {
	// load config
	configPath := "config.json"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
	}
	b, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("read config %s: %v", configPath, err)
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		log.Fatalf("parse config: %v", err)
	}

	cache = newLRU(cfg.Cache.MaxEntries, time.Duration(cfg.Cache.TTLSeconds)*time.Second)
	httpClient = &http.Client{
		Timeout: time.Duration(cfg.BackendTimeoutMs) * time.Millisecond,
		Transport: &http.Transport{
			MaxIdleConns:        16,
			MaxIdleConnsPerHost: 16,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/audio/speech", handleSpeech)
	mux.HandleFunc("/health", handleHealth)

	log.Printf("qwen3tts gateway listening on %s -> backend %s (cache %d entries / %ds, auth=%v)",
		cfg.Listen, cfg.Backend, cfg.Cache.MaxEntries, cfg.Cache.TTLSeconds, cfg.RequireAuth)
	if err := http.ListenAndServe(cfg.Listen, mux); err != nil {
		log.Fatalf("listen: %v", err)
	}
}
