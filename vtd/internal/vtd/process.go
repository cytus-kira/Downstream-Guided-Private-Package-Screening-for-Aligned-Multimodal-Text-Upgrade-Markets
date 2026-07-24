package vtd

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type binaryMarshaler interface{ MarshalBinary() ([]byte, error) }
type binaryUnmarshaler interface{ UnmarshalBinary([]byte) error }

func ensureParent(path string) error { return os.MkdirAll(filepath.Dir(path), 0o755) }

func saveBinary(path string, value binaryMarshaler) error {
	if err := ensureParent(path); err != nil {
		return err
	}
	b, err := value.MarshalBinary()
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600)
}

func loadBinary(path string, value binaryUnmarshaler) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return value.UnmarshalBinary(b)
}

func saveJSON(path string, value any) error {
	if err := ensureParent(path); err != nil {
		return err
	}
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o600)
}

func loadJSON(path string, value any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, value)
}

func writeFile(path string, data []byte, mode os.FileMode) error {
	if err := ensureParent(path); err != nil {
		return err
	}
	return os.WriteFile(path, data, mode)
}

func RootPath(root string, parts ...string) string {
	all := append([]string{root}, parts...)
	return filepath.Join(all...)
}

func InitRun(root string, cfg HEConfig) error {
	for _, d := range []string{"public", "buyer/private", "market/private", "messages", "results"} {
		if err := os.MkdirAll(RootPath(root, d), 0o755); err != nil {
			return err
		}
	}
	return saveJSON(RootPath(root, "config.json"), cfg)
}

func LoadConfig(root string) (HEConfig, ckks.Parameters, error) {
	var cfg HEConfig
	if err := loadJSON(RootPath(root, "config.json"), &cfg); err != nil {
		return cfg, ckks.Parameters{}, err
	}
	params, err := cfg.Parameters()
	return cfg, params, err
}

func keyedCRS(label string) (sampling.PRNG, error) {
	return sampling.NewKeyedPRNG([]byte("ball-market-vtd-crs-v1|" + label))
}

func publicCRP(params ckks.Parameters) (multiparty.PublicKeyGenCRP, error) {
	crs, err := keyedCRS("ckg")
	if err != nil {
		return multiparty.PublicKeyGenCRP{}, err
	}
	return multiparty.NewPublicKeyGenProtocol(params).SampleCRP(crs), nil
}

func PartyRoundOne(root, role string) error {
	if role != "buyer" && role != "market" {
		return fmt.Errorf("invalid role %q", role)
	}
	cfg, params, err := LoadConfig(root)
	if err != nil {
		return err
	}
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crp, err := publicCRP(params)
	if err != nil {
		return err
	}
	ckgShare := ckg.AllocateShare()
	ckg.GenShare(sk, crp, &ckgShare)
	errShare := ckgError(params, sk, crp, ckgShare)

	rkgCRS, err := keyedCRS("rkg")
	if err != nil {
		return err
	}
	rkg := multiparty.NewRelinearizationKeyGenProtocol(params)
	rkgCRP := rkg.SampleCRP(rkgCRS)
	eph, round1, _ := rkg.AllocateShare()
	rkg.GenShareRoundOne(sk, rkgCRP, eph, &round1)

	priv := RootPath(root, role, "private")
	pub := RootPath(root, "public")
	if err := saveBinary(RootPath(priv, "secret.bin"), sk); err != nil {
		return err
	}
	if err := saveBinary(RootPath(priv, "ckg_error.bin"), errShare); err != nil {
		return err
	}
	if err := saveBinary(RootPath(priv, "rkg_ephemeral.bin"), eph); err != nil {
		return err
	}
	if err := saveBinary(RootPath(pub, role+"_ckg_share.bin"), ckgShare); err != nil {
		return err
	}
	if err := saveBinary(RootPath(pub, role+"_rkg_round1.bin"), round1); err != nil {
		return err
	}

	if role == "buyer" {
		policyPub, policyPriv, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}
		if err := writeFile(RootPath(priv, "policy_signing.key"), policyPriv, 0o600); err != nil {
			return err
		}
		if err := writeFile(RootPath(pub, "buyer_policy_public.key"), policyPub, 0o644); err != nil {
			return err
		}
	} else {
		secretOrd := ordinaryPoly(params, sk.Value.Q, cfg.OutputLevel)
		errorOrd := ordinaryPoly(params, errShare.Value.Q, cfg.OutputLevel)
		salt, err := randomSaltElement()
		if err != nil {
			return err
		}
		commitment, err := poseidonCommit(flattenPoly(secretOrd, cfg.OutputLevel), flattenPoly(errorOrd, cfg.OutputLevel), salt)
		if err != nil {
			return err
		}
		if err := writeFile(RootPath(priv, "commitment_salt.txt"), []byte(salt.String()+"\n"), 0o600); err != nil {
			return err
		}
		registration := map[string]any{
			"commitment":        commitment.String(),
			"commitment_scheme": CommitmentID,
			"registered_level":  cfg.OutputLevel,
			"parameter_digest":  cfg.Digest(),
			"created_utc":       time.Now().UTC().Format(time.RFC3339Nano),
		}
		if err := saveJSON(RootPath(pub, "market_registration.json"), registration); err != nil {
			return err
		}
	}
	return nil
}

func CoordinatorRoundOne(root string) error {
	_, params, err := LoadConfig(root)
	if err != nil {
		return err
	}
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	buyerCKG, marketCKG := ckg.AllocateShare(), ckg.AllocateShare()
	if err := loadBinary(RootPath(root, "public", "buyer_ckg_share.bin"), &buyerCKG); err != nil {
		return err
	}
	if err := loadBinary(RootPath(root, "public", "market_ckg_share.bin"), &marketCKG); err != nil {
		return err
	}
	combined := ckg.AllocateShare()
	ckg.AggregateShares(buyerCKG, marketCKG, &combined)
	crp, err := publicCRP(params)
	if err != nil {
		return err
	}
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(combined, crp, pk)
	if err := saveBinary(RootPath(root, "public", "collective_public_key.bin"), pk); err != nil {
		return err
	}

	rkg := multiparty.NewRelinearizationKeyGenProtocol(params)
	_, buyerR1, _ := rkg.AllocateShare()
	_, marketR1, _ := rkg.AllocateShare()
	if err := loadBinary(RootPath(root, "public", "buyer_rkg_round1.bin"), &buyerR1); err != nil {
		return err
	}
	if err := loadBinary(RootPath(root, "public", "market_rkg_round1.bin"), &marketR1); err != nil {
		return err
	}
	_, combinedR1, _ := rkg.AllocateShare()
	rkg.AggregateShares(buyerR1, marketR1, &combinedR1)
	return saveBinary(RootPath(root, "public", "rkg_round1_combined.bin"), combinedR1)
}

func PartyRoundTwo(root, role string) error {
	if role != "buyer" && role != "market" {
		return fmt.Errorf("invalid role %q", role)
	}
	_, params, err := LoadConfig(root)
	if err != nil {
		return err
	}
	sk := rlwe.NewSecretKey(params)
	eph := rlwe.NewSecretKey(params)
	if err := loadBinary(RootPath(root, role, "private", "secret.bin"), sk); err != nil {
		return err
	}
	if err := loadBinary(RootPath(root, role, "private", "rkg_ephemeral.bin"), eph); err != nil {
		return err
	}
	rkg := multiparty.NewRelinearizationKeyGenProtocol(params)
	_, combinedR1, round2 := rkg.AllocateShare()
	if err := loadBinary(RootPath(root, "public", "rkg_round1_combined.bin"), &combinedR1); err != nil {
		return err
	}
	rkg.GenShareRoundTwo(eph, sk, combinedR1, &round2)
	return saveBinary(RootPath(root, "public", role+"_rkg_round2.bin"), round2)
}

func CoordinatorFinalize(root string) error {
	_, params, err := LoadConfig(root)
	if err != nil {
		return err
	}
	rkg := multiparty.NewRelinearizationKeyGenProtocol(params)
	_, combinedR1, combinedR2 := rkg.AllocateShare()
	_, buyerR2, _ := rkg.AllocateShare()
	_, marketR2, _ := rkg.AllocateShare()
	if err := loadBinary(RootPath(root, "public", "rkg_round1_combined.bin"), &combinedR1); err != nil {
		return err
	}
	if err := loadBinary(RootPath(root, "public", "buyer_rkg_round2.bin"), &buyerR2); err != nil {
		return err
	}
	if err := loadBinary(RootPath(root, "public", "market_rkg_round2.bin"), &marketR2); err != nil {
		return err
	}
	rkg.AggregateShares(buyerR2, marketR2, &combinedR2)
	rlk := rlwe.NewRelinearizationKey(params)
	rkg.GenRelinearizationKey(combinedR1, combinedR2, rlk)
	return saveBinary(RootPath(root, "public", "collective_relinearization_key.bin"), rlk)
}

type PackageFixture struct {
	Dataset    string      `json:"dataset"`
	Seed       int         `json:"seed"`
	SourceNPZ  string      `json:"source_npz"`
	PackageIDs []int       `json:"package_ids"`
	Features   [][]float64 `json:"features"`
	Members    [][]int     `json:"members"`
	RowsDigest []string    `json:"rows_digest"`
}

type EncryptedInputMeta struct {
	Dataset      string   `json:"dataset"`
	Seed         int      `json:"seed"`
	SourceNPZ    string   `json:"source_npz"`
	PackageIDs   []int    `json:"package_ids"`
	ActiveSlots  int      `json:"active_slots"`
	FeatureDim   int      `json:"feature_dim"`
	FeatureFiles []string `json:"feature_files"`
	NormFile     string   `json:"norm_file"`
}

func MarketEncrypt(root, fixturePath string) error {
	_, params, err := LoadConfig(root)
	if err != nil {
		return err
	}
	var fixture PackageFixture
	if err := loadJSON(fixturePath, &fixture); err != nil {
		return err
	}
	pk := rlwe.NewPublicKey(params)
	if err := loadBinary(RootPath(root, "public", "collective_public_key.bin"), pk); err != nil {
		return err
	}
	input, err := EncryptPackageSummaries(params, pk, fixture.Features)
	if err != nil {
		return err
	}
	inputDir := RootPath(root, "messages", "encrypted_package_input")
	meta := EncryptedInputMeta{Dataset: fixture.Dataset, Seed: fixture.Seed, SourceNPZ: fixture.SourceNPZ, PackageIDs: fixture.PackageIDs, ActiveSlots: input.ActiveSlots, FeatureDim: input.FeatureDim}
	for i, ct := range input.FeatureCiphertexts {
		name := fmt.Sprintf("feature_%03d.bin", i)
		if err := saveBinary(RootPath(inputDir, name), ct); err != nil {
			return err
		}
		meta.FeatureFiles = append(meta.FeatureFiles, name)
	}
	meta.NormFile = "negative_norm.bin"
	if err := saveBinary(RootPath(inputDir, meta.NormFile), input.NegativeHalfNorm); err != nil {
		return err
	}
	return saveJSON(RootPath(inputDir, "meta.json"), meta)
}

func loadCiphertext(path string) (*rlwe.Ciphertext, error) {
	ct := new(rlwe.Ciphertext)
	if err := loadBinary(path, ct); err != nil {
		return nil, err
	}
	return ct, nil
}

func BuyerScore(root string, landmarks int) (map[string]any, error) {
	_, params, err := LoadConfig(root)
	if err != nil {
		return nil, err
	}
	rkg := rlwe.NewRelinearizationKey(params)
	if err := loadBinary(RootPath(root, "public", "collective_relinearization_key.bin"), rkg); err != nil {
		return nil, err
	}
	inputDir := RootPath(root, "messages", "encrypted_package_input")
	var meta EncryptedInputMeta
	if err := loadJSON(RootPath(inputDir, "meta.json"), &meta); err != nil {
		return nil, err
	}
	input := &EncryptedPackageInput{ActiveSlots: meta.ActiveSlots, FeatureDim: meta.FeatureDim}
	for _, name := range meta.FeatureFiles {
		ct, err := loadCiphertext(RootPath(inputDir, name))
		if err != nil {
			return nil, err
		}
		input.FeatureCiphertexts = append(input.FeatureCiphertexts, ct)
	}
	input.NegativeHalfNorm, err = loadCiphertext(RootPath(inputDir, meta.NormFile))
	if err != nil {
		return nil, err
	}
	start := time.Now()
	score, err := EvaluateQuarticPackageScorer(params, rkg, input, landmarks)
	if err != nil {
		return nil, err
	}
	computeMS := float64(time.Since(start).Microseconds()) / 1000
	if err := saveBinary(RootPath(root, "messages", "score_ciphertext.bin"), score); err != nil {
		return nil, err
	}
	result := map[string]any{"encrypted_compute_ms": computeMS, "landmarks": landmarks, "level": score.Level(), "active_slots": meta.ActiveSlots, "dataset": meta.Dataset, "seed": meta.Seed}
	if err := saveJSON(RootPath(root, "results", "scorer_run.json"), result); err != nil {
		return nil, err
	}
	return result, nil
}

type PolicyRegistry struct {
	SessionID         string `json:"session_id"`
	OutputID          string `json:"output_id"`
	CiphertextDigest  string `json:"ciphertext_digest"`
	CKKSLevel         int    `json:"ckks_level"`
	ReleasePermission bool   `json:"release_permission"`
	RegistryVersion   string `json:"registry_version"`
	Signer            string `json:"signer"`
}

type BuyerReleaseResult struct {
	Accepted            bool      `json:"accepted"`
	SessionID           string    `json:"session_id"`
	OutputID            string    `json:"output_id"`
	ActiveSlots         int       `json:"active_slots"`
	DecodedScores       []float64 `json:"decoded_scores"`
	ProofVerificationMS float64   `json:"proof_verification_ms"`
	ReconstructionMS    float64   `json:"reconstruction_ms"`
	CKKSDecodingMS      float64   `json:"ckks_decoding_ms"`
	ProofBytes          int64     `json:"proof_bytes"`
	MarketToBuyerBytes  int64     `json:"market_to_buyer_bytes"`
}

type SignedPolicy struct {
	Registry  PolicyRegistry `json:"registry"`
	Signature string         `json:"signature"`
}

type MarketRegistration struct {
	Commitment       string `json:"commitment"`
	CommitmentScheme string `json:"commitment_scheme"`
	RegisteredLevel  int    `json:"registered_level"`
	ParameterDigest  string `json:"parameter_digest"`
	CreatedUTC       string `json:"created_utc"`
}

func ValidateMarketRegistration(root string, cfg HEConfig, statement ReleaseStatement) error {
	var registration MarketRegistration
	if err := loadJSON(RootPath(root, "public", "market_registration.json"), &registration); err != nil {
		return err
	}
	if registration.Commitment != statement.MarketCommitment {
		return fmt.Errorf("release proof commitment does not match registered market commitment")
	}
	if registration.CommitmentScheme != statement.CommitmentScheme || registration.CommitmentScheme != CommitmentID {
		return fmt.Errorf("market commitment scheme mismatch")
	}
	if registration.RegisteredLevel != statement.CKKSLevel {
		return fmt.Errorf("market registration level mismatch")
	}
	if registration.ParameterDigest != statement.ParameterDigest || registration.ParameterDigest != cfg.Digest() {
		return fmt.Errorf("market registration parameter mismatch")
	}
	return nil
}

func (p PolicyRegistry) CanonicalBytes() ([]byte, error) { return json.Marshal(p) }

func BuyerAuthorize(root, sessionID, outputID string, permit bool) error {
	ct, err := loadCiphertext(RootPath(root, "messages", "score_ciphertext.bin"))
	if err != nil {
		return err
	}
	digest, err := binaryDigest("BALL-MARKET-VTD-CT-V1", ct)
	if err != nil {
		return err
	}
	registry := PolicyRegistry{SessionID: sessionID, OutputID: outputID, CiphertextDigest: digest, CKKSLevel: ct.Level(), ReleasePermission: permit, RegistryVersion: "v1", Signer: "buyer"}
	b, err := registry.CanonicalBytes()
	if err != nil {
		return err
	}
	priv, err := os.ReadFile(RootPath(root, "buyer", "private", "policy_signing.key"))
	if err != nil {
		return err
	}
	sig := ed25519.Sign(ed25519.PrivateKey(priv), b)
	return saveJSON(RootPath(root, "messages", "signed_policy.json"), SignedPolicy{Registry: registry, Signature: hex.EncodeToString(sig)})
}

func AttackModifyContribution(root, attack string) error {
	_, params, err := LoadConfig(root)
	if err != nil {
		return err
	}
	ct, err := loadCiphertext(RootPath(root, "messages", "score_ciphertext.bin"))
	if err != nil {
		return err
	}
	sk := rlwe.NewSecretKey(params)
	if err := loadBinary(RootPath(root, "market", "private", "secret.bin"), sk); err != nil {
		return err
	}
	var delta *rlwe.Ciphertext
	switch attack {
	case "wrong-target":
		wrong := ct.CopyNew()
		q := params.Q()[0]
		wrong.Value[1].Coeffs[0][0] = (wrong.Value[1].Coeffs[0][0] + 1) % q
		delta, err = PartialDecryptContribution(params, wrong, sk)
	case "wrong-key":
		fresh := rlwe.NewKeyGenerator(params).GenSecretKeyNew()
		delta, err = PartialDecryptContribution(params, ct, fresh)
	case "malformed-share", "excessive-noise":
		delta, err = loadCiphertext(RootPath(root, "messages", "market_contribution.bin"))
		if err == nil {
			q := params.Q()[0]
			add := uint64(1)
			if attack == "excessive-noise" {
				add = q / 4
			}
			delta.Value[0].Coeffs[0][0] = (delta.Value[0].Coeffs[0][0] + add) % q
		}
	default:
		return fmt.Errorf("unknown contribution attack %q", attack)
	}
	if err != nil {
		return err
	}
	return saveBinary(RootPath(root, "messages", "market_contribution.bin"), delta)
}

func VerifyPolicy(root string, policy SignedPolicy, ct *rlwe.Ciphertext) (string, error) {
	b, err := policy.Registry.CanonicalBytes()
	if err != nil {
		return "", err
	}
	pub, err := os.ReadFile(RootPath(root, "public", "buyer_policy_public.key"))
	if err != nil {
		return "", err
	}
	sig, err := hex.DecodeString(policy.Signature)
	if err != nil {
		return "", err
	}
	if !ed25519.Verify(ed25519.PublicKey(pub), b, sig) {
		return "", fmt.Errorf("invalid policy signature")
	}
	if !policy.Registry.ReleasePermission {
		return "", fmt.Errorf("output is not authorized")
	}
	actual, err := binaryDigest("BALL-MARKET-VTD-CT-V1", ct)
	if err != nil {
		return "", err
	}
	if policy.Registry.CiphertextDigest != actual {
		return "", fmt.Errorf("authorized ciphertext digest mismatch")
	}
	if policy.Registry.CKKSLevel != ct.Level() {
		return "", fmt.Errorf("authorized level mismatch")
	}
	d := sha256Bytes(append([]byte("BALL-MARKET-VTD-POLICY-V1\x00"), b...))
	return hex.EncodeToString(d), nil
}

func sha256Bytes(b []byte) []byte { d := sha256.Sum256(b); return d[:] }

func parseFieldString(text string) (out fr.Element, err error) {
	if _, err = out.SetString(strings.TrimSpace(text)); err != nil {
		return out, fmt.Errorf("invalid field element: %w", err)
	}
	return out, nil
}

func SaveProofArtifacts(root string, artifacts *ProofArtifacts, metrics ProofMetrics) error {
	files := []struct {
		name  string
		value io.WriterTo
	}{
		{"release.r1cs", artifacts.CCS},
		{"release.pk", artifacts.PK},
		{"release.vk", artifacts.VK},
	}
	for _, item := range files {
		path := RootPath(root, "public", "proof", item.name)
		if err := ensureParent(path); err != nil {
			return err
		}
		f, err := os.Create(path)
		if err != nil {
			return err
		}
		_, writeErr := item.value.WriteTo(f)
		closeErr := f.Close()
		if writeErr != nil {
			return writeErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return saveJSON(RootPath(root, "results", "proof_setup.json"), metrics)
}

func LoadProofArtifacts(root string) (*ProofArtifacts, error) {
	values := &ProofArtifacts{CCS: groth16.NewCS(ecc.BN254), PK: groth16.NewProvingKey(ecc.BN254), VK: groth16.NewVerifyingKey(ecc.BN254)}
	items := []struct {
		name  string
		value io.ReaderFrom
	}{{"release.r1cs", values.CCS}, {"release.pk", values.PK}, {"release.vk", values.VK}}
	for _, item := range items {
		f, err := os.Open(RootPath(root, "public", "proof", item.name))
		if err != nil {
			return nil, err
		}
		_, readErr := item.value.ReadFrom(f)
		closeErr := f.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
	}
	return values, nil
}

func ProofSetup(root string) (ProofMetrics, error) {
	cfg, params, err := LoadConfig(root)
	if err != nil {
		return ProofMetrics{}, err
	}
	moduli := append([]uint64(nil), params.Q()[:cfg.OutputLevel+1]...)
	artifacts, metrics, err := SetupProofSystem(params.N(), moduli)
	if err != nil {
		return metrics, err
	}
	return metrics, SaveProofArtifacts(root, artifacts, metrics)
}

func loadMarketMaterial(root string, params ckks.Parameters) (*rlwe.SecretKey, *rlwe.SecretKey, multiparty.PublicKeyGenShare, multiparty.PublicKeyGenCRP, fr.Element, error) {
	sk := rlwe.NewSecretKey(params)
	errShare := rlwe.NewSecretKey(params)
	if err := loadBinary(RootPath(root, "market", "private", "secret.bin"), sk); err != nil {
		return nil, nil, multiparty.PublicKeyGenShare{}, multiparty.PublicKeyGenCRP{}, fr.Element{}, err
	}
	if err := loadBinary(RootPath(root, "market", "private", "ckg_error.bin"), errShare); err != nil {
		return nil, nil, multiparty.PublicKeyGenShare{}, multiparty.PublicKeyGenCRP{}, fr.Element{}, err
	}
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	share := ckg.AllocateShare()
	if err := loadBinary(RootPath(root, "public", "market_ckg_share.bin"), &share); err != nil {
		return nil, nil, share, multiparty.PublicKeyGenCRP{}, fr.Element{}, err
	}
	crp, err := publicCRP(params)
	if err != nil {
		return nil, nil, share, crp, fr.Element{}, err
	}
	saltText, err := os.ReadFile(RootPath(root, "market", "private", "commitment_salt.txt"))
	if err != nil {
		return nil, nil, share, crp, fr.Element{}, err
	}
	salt, err := parseFieldString(string(saltText))
	return sk, errShare, share, crp, salt, err
}

func MarketRelease(root, scorerDigest string) (ProofMetrics, error) {
	cfg, params, err := LoadConfig(root)
	if err != nil {
		return ProofMetrics{}, err
	}
	ct, err := loadCiphertext(RootPath(root, "messages", "score_ciphertext.bin"))
	if err != nil {
		return ProofMetrics{}, err
	}
	var signed SignedPolicy
	if err := loadJSON(RootPath(root, "messages", "signed_policy.json"), &signed); err != nil {
		return ProofMetrics{}, err
	}
	policyDigest, err := VerifyPolicy(root, signed, ct)
	if err != nil {
		return ProofMetrics{}, err
	}
	sk, ckgErr, marketCKG, crp, salt, err := loadMarketMaterial(root, params)
	if err != nil {
		return ProofMetrics{}, err
	}
	start := time.Now()
	delta, err := PartialDecryptContribution(params, ct, sk)
	partialMS := float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return ProofMetrics{}, err
	}
	data, err := BuildReleaseProofData(cfg, params,
		ReleaseStatement{SessionID: signed.Registry.SessionID, OutputID: signed.Registry.OutputID, CKKSLevel: ct.Level(), ScorerDigest: scorerDigest, PolicyDigest: policyDigest},
		ct, delta, ct.Value[1], delta.Value[0], sk.Value.Q, ckgErr.Value.Q, marketCKG, crp, &salt)
	if err != nil {
		return ProofMetrics{}, err
	}
	if err := ValidateMarketRegistration(root, cfg, data.Statement); err != nil {
		return ProofMetrics{}, err
	}
	artifacts, err := LoadProofArtifacts(root)
	if err != nil {
		return ProofMetrics{}, err
	}
	proof, metrics, err := ProveAndVerify(artifacts, data)
	if err != nil {
		return metrics, err
	}
	if err := saveBinary(RootPath(root, "messages", "market_contribution.bin"), delta); err != nil {
		return metrics, err
	}
	if err := writeFile(RootPath(root, "messages", "release.proof"), proof, 0o644); err != nil {
		return metrics, err
	}
	if err := saveJSON(RootPath(root, "messages", "release_statement.json"), data.Statement); err != nil {
		return metrics, err
	}
	metricsMap := map[string]any{"partial_decryption_ms": partialMS, "witness_preparation_ms": metrics.WitnessMS, "proof_generation_ms": metrics.ProveMS, "proof_verification_selfcheck_ms": metrics.VerifyMS, "proof_bytes": metrics.ProofBytes, "constraints": metrics.Constraints, "public_variables": metrics.PublicVariables}
	if err := saveJSON(RootPath(root, "results", "market_release.json"), metricsMap); err != nil {
		return metrics, err
	}
	return metrics, nil
}

func BuyerVerifyAndDecode(root string) (BuyerReleaseResult, error) {
	result := BuyerReleaseResult{}
	cfg, params, err := LoadConfig(root)
	if err != nil {
		return result, err
	}
	ct, err := loadCiphertext(RootPath(root, "messages", "score_ciphertext.bin"))
	if err != nil {
		return result, err
	}
	delta, err := loadCiphertext(RootPath(root, "messages", "market_contribution.bin"))
	if err != nil {
		return result, err
	}
	var statement ReleaseStatement
	if err := loadJSON(RootPath(root, "messages", "release_statement.json"), &statement); err != nil {
		return result, err
	}
	var signed SignedPolicy
	if err := loadJSON(RootPath(root, "messages", "signed_policy.json"), &signed); err != nil {
		return result, err
	}
	policyDigest, err := VerifyPolicy(root, signed, ct)
	if err != nil {
		return result, err
	}
	if statement.PolicyDigest != policyDigest || statement.SessionID != signed.Registry.SessionID || statement.OutputID != signed.Registry.OutputID {
		return result, fmt.Errorf("release statement is not bound to the authorized policy")
	}
	if err := ValidateMarketRegistration(root, cfg, statement); err != nil {
		return result, err
	}
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	marketCKG := ckg.AllocateShare()
	if err := loadBinary(RootPath(root, "public", "market_ckg_share.bin"), &marketCKG); err != nil {
		return result, err
	}
	crp, err := publicCRP(params)
	if err != nil {
		return result, err
	}
	publicData, err := RebuildReleasePublicData(cfg, params, statement, ct, delta, ct.Value[1], delta.Value[0], marketCKG, crp)
	if err != nil {
		return result, err
	}
	artifacts, err := LoadProofArtifacts(root)
	if err != nil {
		return result, err
	}
	proofPath := RootPath(root, "messages", "release.proof")
	proofBytes, err := os.ReadFile(proofPath)
	if err != nil {
		return result, err
	}
	verifyMS, err := VerifyProof(artifacts, publicData, proofBytes)
	if err != nil {
		return result, fmt.Errorf("proof rejected: %w", err)
	}
	start := time.Now()
	releaseCT, err := ApplyMarketContribution(params, ct, delta)
	result.ReconstructionMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return result, err
	}
	buyerSK := rlwe.NewSecretKey(params)
	if err := loadBinary(RootPath(root, "buyer", "private", "secret.bin"), buyerSK); err != nil {
		return result, err
	}
	start = time.Now()
	pt := rlwe.NewDecryptor(params, buyerSK).DecryptNew(releaseCT)
	decoded := make([]complex128, params.MaxSlots())
	if err := ckks.NewEncoder(params).Decode(pt, decoded); err != nil {
		return result, err
	}
	result.CKKSDecodingMS = float64(time.Since(start).Microseconds()) / 1000
	var meta EncryptedInputMeta
	if err := loadJSON(RootPath(root, "messages", "encrypted_package_input", "meta.json"), &meta); err != nil {
		return result, err
	}
	result.DecodedScores = make([]float64, meta.ActiveSlots)
	for i := range result.DecodedScores {
		result.DecodedScores[i] = real(decoded[i])
	}
	proofInfo, _ := os.Stat(proofPath)
	deltaInfo, _ := os.Stat(RootPath(root, "messages", "market_contribution.bin"))
	statementInfo, _ := os.Stat(RootPath(root, "messages", "release_statement.json"))
	result.Accepted = true
	result.SessionID = statement.SessionID
	result.OutputID = statement.OutputID
	result.ActiveSlots = meta.ActiveSlots
	result.ProofVerificationMS = verifyMS
	if proofInfo != nil {
		result.ProofBytes = proofInfo.Size()
		result.MarketToBuyerBytes += proofInfo.Size()
	}
	if deltaInfo != nil {
		result.MarketToBuyerBytes += deltaInfo.Size()
	}
	if statementInfo != nil {
		result.MarketToBuyerBytes += statementInfo.Size()
	}
	if err := saveJSON(RootPath(root, "results", "buyer_release.json"), result); err != nil {
		return result, err
	}
	return result, nil
}

func ScorerDigest(landmarks int) string {
	parts := []string{fmt.Sprintf("quartic-krr|landmarks=%d|sigma2=0.5", landmarks)}
	for _, c := range ExpPoly4 {
		parts = append(parts, strconv.FormatFloat(c, 'g', 17, 64))
	}
	d := sha256.Sum256([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(d[:])
}

func sortedKeys[M ~map[string]V, V any](m M) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
