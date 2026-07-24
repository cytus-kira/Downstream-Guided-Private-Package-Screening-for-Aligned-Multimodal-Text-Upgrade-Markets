package vtd

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestTwoPartyThresholdRoundTrip(t *testing.T) {
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
	contribution, err := PartialDecryptContribution(params, score, market.Secret)
	if err != nil {
		t.Fatal(err)
	}
	releaseCT, err := ApplyMarketContribution(params, score, contribution)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeWithBuyerShare(params, releaseCT, buyer.Secret, len(features))
	if err != nil {
		t.Fatal(err)
	}
	want := PlainQuarticPackageScores(features, 2)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 2e-3 {
			t.Fatalf("slot %d: got %.8f want %.8f", i, got[i], want[i])
		}
	}

	// Neither share alone is a decryption key for the aggregate ciphertext.
	for name, sk := range map[string]*rlwe.SecretKey{"buyer": buyer.Secret, "market": market.Secret} {
		pt := rlwe.NewDecryptor(params, sk).DecryptNew(score)
		decoded := make([]complex128, params.MaxSlots())
		if err := ckks.NewEncoder(params).Decode(pt, decoded); err != nil {
			t.Fatal(err)
		}
		if math.Abs(real(decoded[0])-want[0]) < 0.1 {
			t.Fatalf("%s share unexpectedly decrypted aggregate ciphertext", name)
		}
	}
}
