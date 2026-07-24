package vtd

import (
	"crypto/sha256"
	"testing"
)

func TestGroth16ReleaseProofAndTampering(t *testing.T) {
	cfg := SmokeHEConfig()
	params, err := cfg.Parameters()
	if err != nil {
		t.Fatal(err)
	}
	buyer, market, public, err := GenerateCollectiveKeys(params)
	if err != nil {
		t.Fatal(err)
	}
	features := [][]float64{{0.1, -0.2, 0.3, -0.1}, {0.2, 0.1, -0.2, 0.05}}
	enc, err := EncryptPackageSummaries(params, public.PublicKey, features)
	if err != nil {
		t.Fatal(err)
	}
	score, err := EvaluateQuarticPackageScorer(params, public.RelinKey, enc, 2)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := PartialDecryptContribution(params, score, market.Secret)
	if err != nil {
		t.Fatal(err)
	}
	scorerDigest := sha256.Sum256([]byte("test-scorer"))
	data, err := BuildReleaseProofData(
		cfg, params,
		ReleaseStatement{SessionID: "test-session", OutputID: "score-0", CKKSLevel: score.Level(), ScorerDigest: stringHex(scorerDigest[:]), PolicyDigest: stringHex(scorerDigest[:])},
		score, delta, score.Value[1], delta.Value[0], market.Secret.Value.Q, market.CKGErrorQ.Value.Q,
		public.MarketCKG, public.CRP, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	artifacts, _, err := SetupProofSystem(data.N, data.Moduli)
	if err != nil {
		t.Fatal(err)
	}
	proof, _, err := ProveAndVerify(artifacts, data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyProof(artifacts, data, proof); err != nil {
		t.Fatal(err)
	}
	proof[len(proof)/2] ^= 1
	if _, err := VerifyProof(artifacts, data, proof); err == nil {
		t.Fatal("tampered proof was accepted")
	}
	_ = buyer
}

func stringHex(b []byte) string {
	const digits = "0123456789abcdef"
	out := make([]byte, len(b)*2)
	for i, v := range b {
		out[2*i] = digits[v>>4]
		out[2*i+1] = digits[v&15]
	}
	return string(out)
}
