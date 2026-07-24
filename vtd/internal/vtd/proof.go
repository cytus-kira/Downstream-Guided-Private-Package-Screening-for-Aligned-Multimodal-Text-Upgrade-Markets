package vtd

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"math/bits"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	frposeidon "github.com/consensys/gnark-crypto/ecc/bn254/fr/poseidon2"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	stdposeidon "github.com/consensys/gnark/std/hash/poseidon2"
	"github.com/consensys/gnark/std/rangecheck"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	ProofBackendID = "gnark-v0.15.0-groth16-bn254"
	CommitmentID   = "poseidon2-bn254-merkle-damgard-salted-release-share-v1"
	proofDomain    = "BALL-MARKET-VTD-RELEASE-COMMITMENT-V1"
)

type ReleaseStatement struct {
	ProtocolVersion      string   `json:"protocol_version"`
	SessionID            string   `json:"session_id"`
	OutputID             string   `json:"output_id"`
	CiphertextDigest     string   `json:"ciphertext_digest"`
	ContributionDigest   string   `json:"contribution_digest"`
	CKKSLevel            int      `json:"ckks_level"`
	RingDimension        int      `json:"ring_dimension"`
	RNSModuli            []uint64 `json:"rns_moduli"`
	ScaleBits            int      `json:"scale_bits"`
	ParameterDigest      string   `json:"parameter_digest"`
	ScorerDigest         string   `json:"scorer_digest"`
	PolicyDigest         string   `json:"policy_digest"`
	MarketCKGShareDigest string   `json:"market_ckg_share_digest"`
	MarketCommitment     string   `json:"market_commitment"`
	CommitmentScheme     string   `json:"commitment_scheme"`
	ProofBackend         string   `json:"proof_backend"`
}

func (s ReleaseStatement) CanonicalBytes() ([]byte, error) {
	return json.Marshal(s)
}

func (s ReleaseStatement) Digest() ([32]byte, error) {
	b, err := s.CanonicalBytes()
	if err != nil {
		return [32]byte{}, err
	}
	return sha256.Sum256(append([]byte("BALL-MARKET-VTD-STATEMENT-V1\x00"), b...)), nil
}

func binaryDigest(domain string, value interface{ MarshalBinary() ([]byte, error) }) (string, error) {
	b, err := value.MarshalBinary()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	h.Write([]byte(domain))
	h.Write([]byte{0})
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil)), nil
}

func fieldFromDigest(digest [32]byte) fr.Element {
	var out fr.Element
	out.SetBytes(digest[:])
	return out
}

func domainElement() fr.Element {
	d := sha256.Sum256([]byte(proofDomain))
	return fieldFromDigest(d)
}

func randomSaltElement() (fr.Element, error) {
	var raw [31]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return fr.Element{}, err
	}
	var out fr.Element
	out.SetBytes(raw[:])
	return out, nil
}

func poseidonCommit(secret, ckgError []uint64, salt fr.Element) (fr.Element, error) {
	h := frposeidon.NewMerkleDamgardHasher()
	writeElement := func(e fr.Element) error {
		_, err := h.Write(e.Marshal())
		return err
	}
	if err := writeElement(domainElement()); err != nil {
		return fr.Element{}, err
	}
	if err := writeElement(salt); err != nil {
		return fr.Element{}, err
	}
	for _, values := range [][]uint64{secret, ckgError} {
		for _, v := range values {
			var e fr.Element
			e.SetUint64(v)
			if err := writeElement(e); err != nil {
				return fr.Element{}, err
			}
		}
	}
	var out fr.Element
	out.SetBytes(h.Sum(nil))
	return out, nil
}

func ordinaryPoly(params ckks.Parameters, in ring.Poly, level int) ring.Poly {
	ringQ := params.RingQ().AtLevel(level)
	out := ringQ.NewPoly()
	out.CopyLvl(level, in)
	ringQ.IMForm(out, out)
	return out
}

func flattenPoly(poly ring.Poly, level int) []uint64 {
	n := poly.N()
	out := make([]uint64, 0, (level+1)*n)
	for limb := 0; limb <= level; limb++ {
		out = append(out, poly.Coeffs[limb]...)
	}
	return out
}

type ReleaseProofData struct {
	Statement       ReleaseStatement
	StatementDigest [32]byte
	Commitment      fr.Element
	Salt            fr.Element
	Secret          []uint64
	CKGError        []uint64
	C1              []uint64
	A               []uint64
	Delta           []uint64
	PK              []uint64
	ReleaseQuotient []*big.Int
	PKQuotient      []*big.Int
	Moduli          []uint64
	N               int
	Level           int
}

func BuildReleaseProofData(
	cfg HEConfig,
	params ckks.Parameters,
	statementBase ReleaseStatement,
	ct, contribution interface{ MarshalBinary() ([]byte, error) },
	ctC1, delta ring.Poly,
	marketSKQ, marketErrorQ ring.Poly,
	marketCKG multiparty.PublicKeyGenShare,
	crp multiparty.PublicKeyGenCRP,
	salt *fr.Element,
) (*ReleaseProofData, error) {
	level := statementBase.CKKSLevel
	if level < 0 || level > params.MaxLevel() {
		return nil, fmt.Errorf("invalid release level %d", level)
	}
	ctDigest, err := binaryDigest("BALL-MARKET-VTD-CT-V1", ct)
	if err != nil {
		return nil, err
	}
	deltaDigest, err := binaryDigest("BALL-MARKET-VTD-DELTA-V1", contribution)
	if err != nil {
		return nil, err
	}
	ckgDigest, err := binaryDigest("BALL-MARKET-VTD-CKG-SHARE-V1", marketCKG)
	if err != nil {
		return nil, err
	}
	secretOrd := ordinaryPoly(params, marketSKQ, level)
	errorOrd := ordinaryPoly(params, marketErrorQ, level)
	c1Ord := ordinaryPoly(params, ctC1, level)
	deltaOrd := ordinaryPoly(params, delta, level)
	aOrd := ordinaryPoly(params, crp.Value.Q, level)
	pkOrd := ordinaryPoly(params, marketCKG.Value.Q, level)
	secretFlat := flattenPoly(secretOrd, level)
	errorFlat := flattenPoly(errorOrd, level)

	actualSalt := fr.Element{}
	if salt == nil {
		actualSalt, err = randomSaltElement()
		if err != nil {
			return nil, err
		}
	} else {
		actualSalt.Set(salt)
	}
	commitment, err := poseidonCommit(secretFlat, errorFlat, actualSalt)
	if err != nil {
		return nil, err
	}
	statementBase.ProtocolVersion = "vtd-2of2-v1"
	statementBase.CiphertextDigest = ctDigest
	statementBase.ContributionDigest = deltaDigest
	statementBase.MarketCKGShareDigest = ckgDigest
	statementBase.MarketCommitment = commitment.String()
	statementBase.CommitmentScheme = CommitmentID
	statementBase.ProofBackend = ProofBackendID
	statementBase.ParameterDigest = cfg.Digest()
	statementBase.RingDimension = params.N()
	statementBase.RNSModuli = append([]uint64(nil), params.Q()[:level+1]...)
	statementBase.ScaleBits = cfg.ScaleBits
	statementDigest, err := statementBase.Digest()
	if err != nil {
		return nil, err
	}

	n := params.N()
	count := (level + 1) * n
	c1Flat := flattenPoly(c1Ord, level)
	aFlat := flattenPoly(aOrd, level)
	deltaFlat := flattenPoly(deltaOrd, level)
	pkFlat := flattenPoly(pkOrd, level)
	releaseQ := make([]*big.Int, count)
	pkQ := make([]*big.Int, count)
	for limb := 0; limb <= level; limb++ {
		q := new(big.Int).SetUint64(params.Q()[limb])
		for j := 0; j < n; j++ {
			idx := limb*n + j
			lhsRelease := new(big.Int).Mul(new(big.Int).SetUint64(c1Flat[idx]), new(big.Int).SetUint64(secretFlat[idx]))
			lhsRelease.Add(lhsRelease, q)
			diffRelease := new(big.Int).Sub(lhsRelease, new(big.Int).SetUint64(deltaFlat[idx]))
			releaseQ[idx] = new(big.Int)
			rem := new(big.Int)
			releaseQ[idx].QuoRem(diffRelease, q, rem)
			if rem.Sign() != 0 || releaseQ[idx].Sign() < 0 {
				return nil, fmt.Errorf("release relation failed before proving at limb %d coefficient %d", limb, j)
			}

			lhsPK := new(big.Int).Mul(new(big.Int).SetUint64(aFlat[idx]), new(big.Int).SetUint64(secretFlat[idx]))
			lhsPK.Add(lhsPK, new(big.Int).SetUint64(pkFlat[idx]))
			lhsPK.Add(lhsPK, q)
			diffPK := new(big.Int).Sub(lhsPK, new(big.Int).SetUint64(errorFlat[idx]))
			pkQ[idx] = new(big.Int)
			pkQ[idx].QuoRem(diffPK, q, rem)
			if rem.Sign() != 0 || pkQ[idx].Sign() < 0 {
				return nil, fmt.Errorf("registered-key relation failed before proving at limb %d coefficient %d", limb, j)
			}
		}
	}
	return &ReleaseProofData{
		Statement:       statementBase,
		StatementDigest: statementDigest,
		Commitment:      commitment,
		Salt:            actualSalt,
		Secret:          secretFlat,
		CKGError:        errorFlat,
		C1:              c1Flat,
		A:               aFlat,
		Delta:           deltaFlat,
		PK:              pkFlat,
		ReleaseQuotient: releaseQ,
		PKQuotient:      pkQ,
		Moduli:          append([]uint64(nil), params.Q()[:level+1]...),
		N:               n,
		Level:           level,
	}, nil
}

// RebuildReleasePublicData is used by the buyer/verifier. It derives every
// Groth16 public input from the exact ciphertext, submitted contribution,
// registered CKG share, CRP, and canonical statement. It never accepts witness
// material or a prover-supplied public-witness blob.
func RebuildReleasePublicData(
	cfg HEConfig,
	params ckks.Parameters,
	statement ReleaseStatement,
	ct, contribution interface{ MarshalBinary() ([]byte, error) },
	ctC1, delta ring.Poly,
	marketCKG multiparty.PublicKeyGenShare,
	crp multiparty.PublicKeyGenCRP,
) (*ReleaseProofData, error) {
	level := statement.CKKSLevel
	if level < 0 || level > params.MaxLevel() {
		return nil, fmt.Errorf("invalid release level %d", level)
	}
	ctDigest, err := binaryDigest("BALL-MARKET-VTD-CT-V1", ct)
	if err != nil {
		return nil, err
	}
	deltaDigest, err := binaryDigest("BALL-MARKET-VTD-DELTA-V1", contribution)
	if err != nil {
		return nil, err
	}
	ckgDigest, err := binaryDigest("BALL-MARKET-VTD-CKG-SHARE-V1", marketCKG)
	if err != nil {
		return nil, err
	}
	if statement.CiphertextDigest != ctDigest {
		return nil, fmt.Errorf("statement ciphertext digest mismatch")
	}
	if statement.ContributionDigest != deltaDigest {
		return nil, fmt.Errorf("statement contribution digest mismatch")
	}
	if statement.MarketCKGShareDigest != ckgDigest {
		return nil, fmt.Errorf("statement CKG share digest mismatch")
	}
	if statement.ParameterDigest != cfg.Digest() || statement.RingDimension != params.N() {
		return nil, fmt.Errorf("statement CKKS parameter mismatch")
	}
	if statement.ProofBackend != ProofBackendID || statement.CommitmentScheme != CommitmentID {
		return nil, fmt.Errorf("statement proof/commitment backend mismatch")
	}
	if len(statement.RNSModuli) != level+1 {
		return nil, fmt.Errorf("statement RNS limb count mismatch")
	}
	for i, q := range params.Q()[:level+1] {
		if statement.RNSModuli[i] != q {
			return nil, fmt.Errorf("statement modulus mismatch at limb %d", i)
		}
	}
	var commitment fr.Element
	if _, err := commitment.SetString(statement.MarketCommitment); err != nil {
		return nil, fmt.Errorf("invalid commitment: %w", err)
	}
	digest, err := statement.Digest()
	if err != nil {
		return nil, err
	}
	c1Ord := ordinaryPoly(params, ctC1, level)
	deltaOrd := ordinaryPoly(params, delta, level)
	aOrd := ordinaryPoly(params, crp.Value.Q, level)
	pkOrd := ordinaryPoly(params, marketCKG.Value.Q, level)
	n := params.N()
	return &ReleaseProofData{
		Statement: statement, StatementDigest: digest, Commitment: commitment,
		C1: flattenPoly(c1Ord, level), A: flattenPoly(aOrd, level),
		Delta: flattenPoly(deltaOrd, level), PK: flattenPoly(pkOrd, level),
		Moduli: append([]uint64(nil), params.Q()[:level+1]...), N: n, Level: level,
	}, nil
}

type ReleaseCircuit struct {
	Commitment frontend.Variable   `gnark:",public"`
	MetaDigest frontend.Variable   `gnark:",public"`
	C1         []frontend.Variable `gnark:",public"`
	A          []frontend.Variable `gnark:",public"`
	Delta      []frontend.Variable `gnark:",public"`
	PK         []frontend.Variable `gnark:",public"`

	Salt            frontend.Variable
	Secret          []frontend.Variable
	CKGError        []frontend.Variable
	ReleaseQuotient []frontend.Variable
	PKQuotient      []frontend.Variable

	Moduli []uint64 `gnark:"-"`
	N      int      `gnark:"-"`
}

func NewReleaseCircuit(n int, moduli []uint64) *ReleaseCircuit {
	count := n * len(moduli)
	return &ReleaseCircuit{
		C1:              make([]frontend.Variable, count),
		A:               make([]frontend.Variable, count),
		Delta:           make([]frontend.Variable, count),
		PK:              make([]frontend.Variable, count),
		Secret:          make([]frontend.Variable, count),
		CKGError:        make([]frontend.Variable, count),
		ReleaseQuotient: make([]frontend.Variable, count),
		PKQuotient:      make([]frontend.Variable, count),
		Moduli:          append([]uint64(nil), moduli...),
		N:               n,
	}
}

func (c *ReleaseCircuit) Define(api frontend.API) error {
	h, err := stdposeidon.New(api)
	if err != nil {
		return err
	}
	domain := domainElement()
	h.Write(domain.String(), c.Salt)
	h.Write(c.Secret...)
	h.Write(c.CKGError...)
	api.AssertIsEqual(h.Sum(), c.Commitment)
	api.AssertIsDifferent(c.MetaDigest, 0)

	rc := rangecheck.New(api)
	for limb, modulus := range c.Moduli {
		for j := 0; j < c.N; j++ {
			idx := limb*c.N + j
			lhsRelease := api.Add(api.Mul(c.C1[idx], c.Secret[idx]), modulus)
			api.AssertIsEqual(lhsRelease, api.Add(c.Delta[idx], api.Mul(modulus, c.ReleaseQuotient[idx])))
			lhsPK := api.Add(api.Mul(c.A[idx], c.Secret[idx]), c.PK[idx], modulus)
			api.AssertIsEqual(lhsPK, api.Add(c.CKGError[idx], api.Mul(modulus, c.PKQuotient[idx])))
			qBits := bits.Len64(modulus) + 1
			rc.Check(c.ReleaseQuotient[idx], qBits)
			rc.Check(c.PKQuotient[idx], qBits)
		}
	}
	return nil
}

func (d *ReleaseProofData) Assignment(publicOnly bool) *ReleaseCircuit {
	a := NewReleaseCircuit(d.N, d.Moduli)
	a.Commitment = d.Commitment.String()
	meta := fieldFromDigest(d.StatementDigest)
	if meta.IsZero() {
		meta.SetOne()
	}
	a.MetaDigest = meta.String()
	for i := range d.C1 {
		a.C1[i] = d.C1[i]
		a.A[i] = d.A[i]
		a.Delta[i] = d.Delta[i]
		a.PK[i] = d.PK[i]
	}
	if publicOnly {
		return a
	}
	a.Salt = d.Salt.String()
	for i := range d.Secret {
		a.Secret[i] = d.Secret[i]
		a.CKGError[i] = d.CKGError[i]
	}
	for i := range d.ReleaseQuotient {
		a.ReleaseQuotient[i] = d.ReleaseQuotient[i]
		a.PKQuotient[i] = d.PKQuotient[i]
	}
	return a
}

type ProofArtifacts struct {
	CCS constraint.ConstraintSystem
	PK  groth16.ProvingKey
	VK  groth16.VerifyingKey
}

type ProofMetrics struct {
	CompileMS       float64
	SetupMS         float64
	WitnessMS       float64
	ProveMS         float64
	VerifyMS        float64
	ProofBytes      int
	Constraints     int
	PublicVariables int
}

func SetupProofSystem(n int, moduli []uint64) (*ProofArtifacts, ProofMetrics, error) {
	metrics := ProofMetrics{}
	start := time.Now()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, NewReleaseCircuit(n, moduli))
	metrics.CompileMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return nil, metrics, err
	}
	metrics.Constraints = ccs.GetNbConstraints()
	metrics.PublicVariables = ccs.GetNbPublicVariables()
	start = time.Now()
	pk, vk, err := groth16.Setup(ccs)
	metrics.SetupMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return nil, metrics, err
	}
	return &ProofArtifacts{CCS: ccs, PK: pk, VK: vk}, metrics, nil
}

func ProveAndVerify(artifacts *ProofArtifacts, data *ReleaseProofData) ([]byte, ProofMetrics, error) {
	metrics := ProofMetrics{Constraints: artifacts.CCS.GetNbConstraints(), PublicVariables: artifacts.CCS.GetNbPublicVariables()}
	start := time.Now()
	witness, err := frontend.NewWitness(data.Assignment(false), ecc.BN254.ScalarField())
	metrics.WitnessMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return nil, metrics, err
	}
	publicWitness, err := witness.Public()
	if err != nil {
		return nil, metrics, err
	}
	start = time.Now()
	proof, err := groth16.Prove(artifacts.CCS, artifacts.PK, witness)
	metrics.ProveMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return nil, metrics, err
	}
	var buf bytes.Buffer
	if _, err := proof.WriteTo(&buf); err != nil {
		return nil, metrics, err
	}
	metrics.ProofBytes = buf.Len()
	start = time.Now()
	err = groth16.Verify(proof, artifacts.VK, publicWitness)
	metrics.VerifyMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		return nil, metrics, err
	}
	return buf.Bytes(), metrics, nil
}

func VerifyProof(artifacts *ProofArtifacts, data *ReleaseProofData, proofBytes []byte) (float64, error) {
	proof := groth16.NewProof(ecc.BN254)
	if _, err := proof.ReadFrom(bytes.NewReader(proofBytes)); err != nil {
		return 0, err
	}
	publicWitness, err := frontend.NewWitness(data.Assignment(true), ecc.BN254.ScalarField(), frontend.PublicOnly())
	if err != nil {
		return 0, err
	}
	start := time.Now()
	err = groth16.Verify(proof, artifacts.VK, publicWitness)
	return float64(time.Since(start).Microseconds()) / 1000, err
}
