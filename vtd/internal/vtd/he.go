package vtd

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

const (
	BackendID = "lattigo-v6.2.0-ckks-multiparty"
	ScaleBits = 32
)

var (
	DataModulusBits = []int{45, 32, 32, 32, 32}
	SpecialModulusBits = []int{45}
	ExpPoly4 = []float64{
		0.9963358096138180,
		0.9534382874063090,
		0.3987763197687612,
		0.0819512647685308,
		0.0066488054692884,
	}
)

type HEConfig struct {
	LogN       int   `json:"log_n"`
	LogQ       []int `json:"log_q"`
	LogP       []int `json:"log_p"`
	ScaleBits  int   `json:"scale_bits"`
	OutputLevel int  `json:"output_level"`
}

func ProductionHEConfig() HEConfig {
	return HEConfig{
		LogN: 13,
		LogQ: append([]int(nil), DataModulusBits...),
		LogP: append([]int(nil), SpecialModulusBits...),
		ScaleBits: ScaleBits,
		OutputLevel: 0,
	}
}

func SmokeHEConfig() HEConfig {
	return HEConfig{
		LogN: 10,
		LogQ: append([]int(nil), DataModulusBits...),
		LogP: append([]int(nil), SpecialModulusBits...),
		ScaleBits: ScaleBits,
		OutputLevel: 0,
	}
}

func (c HEConfig) Parameters() (ckks.Parameters, error) {
	return ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: c.LogN,
		LogQ: append([]int(nil), c.LogQ...),
		LogP: append([]int(nil), c.LogP...),
		LogDefaultScale: c.ScaleBits,
	})
}

func (c HEConfig) Digest() string {
	h := sha256.New()
	fmt.Fprintf(h, "vtd-he-v1|backend=%s|logN=%d|logQ=%v|logP=%v|scale=%d|out=%d",
		BackendID, c.LogN, c.LogQ, c.LogP, c.ScaleBits, c.OutputLevel)
	return hex.EncodeToString(h.Sum(nil))
}

type PartyKeyMaterial struct {
	Secret   *rlwe.SecretKey
	CKGShare multiparty.PublicKeyGenShare
	CKGErrorQ *rlwe.SecretKey
	RKGEPH   *rlwe.SecretKey
	RKGOne   multiparty.RelinearizationKeyGenShare
	RKGTwo   multiparty.RelinearizationKeyGenShare
}

type CollectiveKeys struct {
	PublicKey *rlwe.PublicKey
	RelinKey  *rlwe.RelinearizationKey
	CRP       multiparty.PublicKeyGenCRP
	BuyerCKG  multiparty.PublicKeyGenShare
	MarketCKG multiparty.PublicKeyGenShare
}

// GenerateCollectiveKeys executes Lattigo's native two-party CKG and two-round
// RKG protocols. The aggregate secret key is never materialized.
func GenerateCollectiveKeys(params ckks.Parameters) (buyer, market PartyKeyMaterial, public CollectiveKeys, err error) {
	crs, err := sampling.NewKeyedPRNG([]byte("ball-market-vtd-ckg-rkg-crs-v1"))
	if err != nil {
		return buyer, market, public, err
	}
	kgen := rlwe.NewKeyGenerator(params)
	buyer.Secret = kgen.GenSecretKeyNew()
	market.Secret = kgen.GenSecretKeyNew()

	ckg := multiparty.NewPublicKeyGenProtocol(params)
	crp := ckg.SampleCRP(crs)
	buyer.CKGShare = ckg.AllocateShare()
	market.CKGShare = ckg.AllocateShare()
	ckg.GenShare(buyer.Secret, crp, &buyer.CKGShare)
	ckg.GenShare(market.Secret, crp, &market.CKGShare)

	combinedCKG := ckg.AllocateShare()
	ckg.AggregateShares(buyer.CKGShare, market.CKGShare, &combinedCKG)
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(combinedCKG, crp, pk)

	// Record e_i = pk_i + a*s_i in the backend's NTT/Montgomery form. It is
	// committed as proof witness material, never sent to the buyer/verifier.
	buyer.CKGErrorQ = ckgError(params, buyer.Secret, crp, buyer.CKGShare)
	market.CKGErrorQ = ckgError(params, market.Secret, crp, market.CKGShare)

	rkg := multiparty.NewRelinearizationKeyGenProtocol(params)
	buyer.RKGEPH, buyer.RKGOne, buyer.RKGTwo = rkg.AllocateShare()
	market.RKGEPH, market.RKGOne, market.RKGTwo = rkg.AllocateShare()
	rkgCRP := rkg.SampleCRP(crs)
	rkg.GenShareRoundOne(buyer.Secret, rkgCRP, buyer.RKGEPH, &buyer.RKGOne)
	rkg.GenShareRoundOne(market.Secret, rkgCRP, market.RKGEPH, &market.RKGOne)
	_, rkgCombinedOne, rkgCombinedTwo := rkg.AllocateShare()
	rkg.AggregateShares(buyer.RKGOne, market.RKGOne, &rkgCombinedOne)
	rkg.GenShareRoundTwo(buyer.RKGEPH, buyer.Secret, rkgCombinedOne, &buyer.RKGTwo)
	rkg.GenShareRoundTwo(market.RKGEPH, market.Secret, rkgCombinedOne, &market.RKGTwo)
	rkg.AggregateShares(buyer.RKGTwo, market.RKGTwo, &rkgCombinedTwo)
	rlk := rlwe.NewRelinearizationKey(params)
	rkg.GenRelinearizationKey(rkgCombinedOne, rkgCombinedTwo, rlk)

	public = CollectiveKeys{
		PublicKey: pk,
		RelinKey: rlk,
		CRP: crp,
		BuyerCKG: buyer.CKGShare,
		MarketCKG: market.CKGShare,
	}
	return
}

func ckgError(params ckks.Parameters, sk *rlwe.SecretKey, crp multiparty.PublicKeyGenCRP, share multiparty.PublicKeyGenShare) *rlwe.SecretKey {
	out := rlwe.NewSecretKey(params)
	out.Value.Copy(share.Value)
	params.RingQP().MulCoeffsMontgomeryThenAdd(sk.Value, crp.Value, out.Value)
	return out
}

type EncryptedPackageInput struct {
	FeatureCiphertexts []*rlwe.Ciphertext
	NegativeHalfNorm   *rlwe.Ciphertext
	ActiveSlots        int
	FeatureDim         int
}

func EncryptPackageSummaries(params ckks.Parameters, pk *rlwe.PublicKey, features [][]float64) (*EncryptedPackageInput, error) {
	if len(features) == 0 {
		return nil, fmt.Errorf("empty package feature matrix")
	}
	dim := len(features[0])
	if dim == 0 {
		return nil, fmt.Errorf("zero-dimensional package features")
	}
	if len(features) > params.MaxSlots() {
		return nil, fmt.Errorf("%d packages exceed %d CKKS slots", len(features), params.MaxSlots())
	}
	for i := range features {
		if len(features[i]) != dim {
			return nil, fmt.Errorf("ragged feature matrix at row %d", i)
		}
	}
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	out := &EncryptedPackageInput{
		FeatureCiphertexts: make([]*rlwe.Ciphertext, dim),
		ActiveSlots: len(features),
		FeatureDim: dim,
	}
	for d := 0; d < dim; d++ {
		values := make([]complex128, params.MaxSlots())
		for i := range features {
			values[i] = complex(features[i][d], 0)
		}
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		if err := encoder.Encode(values, pt); err != nil {
			return nil, err
		}
		ct, err := encryptor.EncryptNew(pt)
		if err != nil {
			return nil, err
		}
		out.FeatureCiphertexts[d] = ct
	}
	norm := make([]complex128, params.MaxSlots())
	for i := range features {
		var n2 float64
		for _, v := range features[i] {
			n2 += v * v
		}
		norm[i] = complex(-n2, 0) // -||x||^2; scorer divides by 2 sigma^2.
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(norm, pt); err != nil {
		return nil, err
	}
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		return nil, err
	}
	out.NegativeHalfNorm = ct
	return out, nil
}

func deterministicUnitReference(index, dim int) []float64 {
	ref := make([]float64, dim)
	var n2 float64
	for d := 0; d < dim; d++ {
		v := math.Sin(0.017*float64((index+1)*(d+1))) + 0.5*math.Cos(0.031*float64((index+3)*(d+1)))
		ref[d] = v
		n2 += v * v
	}
	inv := 1.0 / math.Sqrt(math.Max(n2, 1e-18))
	for i := range ref {
		ref[i] *= inv
	}
	return ref
}

func mulRelinRescale(eval *ckks.Evaluator, a, b *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	out, err := eval.MulRelinNew(a, b)
	if err != nil {
		return nil, err
	}
	if err = eval.Rescale(out, out); err != nil {
		return nil, err
	}
	return out, nil
}

func mulScalarRescale(eval *ckks.Evaluator, a *rlwe.Ciphertext, scalar float64) (*rlwe.Ciphertext, error) {
	out, err := eval.MulRelinNew(a, scalar)
	if err != nil {
		return nil, err
	}
	if err = eval.Rescale(out, out); err != nil {
		return nil, err
	}
	return out, nil
}

func alignLevel(eval *ckks.Evaluator, ct *rlwe.Ciphertext, level int) {
	if ct.Level() > level {
		eval.DropLevel(ct, ct.Level()-level)
	}
}

func polynomial4(eval *ckks.Evaluator, x *rlwe.Ciphertext, coeff []float64, weight float64) (*rlwe.Ciphertext, error) {
	if len(coeff) != 5 {
		return nil, fmt.Errorf("degree-4 polynomial needs five coefficients")
	}
	x2, err := mulRelinRescale(eval, x, x)
	if err != nil {
		return nil, err
	}
	x4, err := mulRelinRescale(eval, x2, x2)
	if err != nil {
		return nil, err
	}
	xAtX2 := x.CopyNew()
	alignLevel(eval, xAtX2, x2.Level())
	x3, err := mulRelinRescale(eval, xAtX2, x2)
	if err != nil {
		return nil, err
	}
	terms := make([]*rlwe.Ciphertext, 0, 4)
	for _, spec := range []struct{ ct *rlwe.Ciphertext; c float64 }{
		{x, coeff[1]}, {x2, coeff[2]}, {x3, coeff[3]}, {x4, coeff[4]},
	} {
		if math.Abs(weight*spec.c) <= 1e-15 {
			continue
		}
		t, err := mulScalarRescale(eval, spec.ct, weight*spec.c)
		if err != nil {
			return nil, err
		}
		terms = append(terms, t)
	}
	if len(terms) == 0 {
		return nil, fmt.Errorf("empty encrypted polynomial")
	}
	target := terms[len(terms)-1].Level()
	for _, t := range terms {
		alignLevel(eval, t, target)
	}
	result := terms[0]
	for i := 1; i < len(terms); i++ {
		if err := eval.Add(result, terms[i], result); err != nil {
			return nil, err
		}
	}
	if err := eval.Add(result, weight*coeff[0], result); err != nil {
		return nil, err
	}
	return result, nil
}

func EvaluateQuarticPackageScorer(params ckks.Parameters, rlk *rlwe.RelinearizationKey, input *EncryptedPackageInput, landmarks int) (*rlwe.Ciphertext, error) {
	if landmarks <= 0 {
		return nil, fmt.Errorf("landmarks must be positive")
	}
	eval := ckks.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(rlk))
	const sigma2 = 0.5
	var score *rlwe.Ciphertext
	for r := 0; r < landmarks; r++ {
		ref := deterministicUnitReference(r, input.FeatureDim)
		var u *rlwe.Ciphertext
		for d, feature := range input.FeatureCiphertexts {
			term, err := mulScalarRescale(eval, feature, ref[d]/sigma2)
			if err != nil {
				return nil, err
			}
			if u == nil {
				u = term
			} else if err := eval.Add(u, term, u); err != nil {
				return nil, err
			}
		}
		norm := input.NegativeHalfNorm.CopyNew()
		alignLevel(eval, norm, u.Level())
		if err := eval.Add(u, norm, u); err != nil {
			return nil, err
		}
		if err := eval.Add(u, -1.0, u); err != nil {
			return nil, err
		}
		alpha := math.Sin(0.013*float64(r+1)) / math.Sqrt(float64(landmarks))
		term, err := polynomial4(eval, u, ExpPoly4, alpha)
		if err != nil {
			return nil, err
		}
		if score == nil {
			score = term
		} else {
			alignLevel(eval, score, term.Level())
			if err := eval.Add(score, term, score); err != nil {
				return nil, err
			}
		}
	}
	return score, nil
}

func PartialDecryptContribution(params ckks.Parameters, ct *rlwe.Ciphertext, sk *rlwe.SecretKey) (*rlwe.Ciphertext, error) {
	if ct.Degree() != 1 || !ct.IsNTT {
		return nil, fmt.Errorf("expected degree-1 NTT CKKS ciphertext")
	}
	out := ct.CopyNew()
	level := ct.Level()
	ringQ := params.RingQ().AtLevel(level)
	ringQ.MulCoeffsMontgomery(ct.Value[1], sk.Value.Q, out.Value[0])
	// The returned object transports delta in Value[0]; Value[1] is zeroed so it
	// cannot accidentally be interpreted as a decryptable ciphertext.
	out.Value[1].Zero()
	return out, nil
}

func ApplyMarketContribution(params ckks.Parameters, ct, contribution *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	if ct.Level() != contribution.Level() {
		return nil, fmt.Errorf("level mismatch: ciphertext=%d contribution=%d", ct.Level(), contribution.Level())
	}
	out := ct.CopyNew()
	params.RingQ().AtLevel(ct.Level()).Add(ct.Value[0], contribution.Value[0], out.Value[0])
	return out, nil
}

func DecodeWithBuyerShare(params ckks.Parameters, ct *rlwe.Ciphertext, buyerSK *rlwe.SecretKey, slots int) ([]float64, error) {
	pt := rlwe.NewDecryptor(params, buyerSK).DecryptNew(ct)
	decoded := make([]complex128, params.MaxSlots())
	if err := ckks.NewEncoder(params).Decode(pt, decoded); err != nil {
		return nil, err
	}
	if slots > len(decoded) {
		slots = len(decoded)
	}
	out := make([]float64, slots)
	for i := range out {
		out[i] = real(decoded[i])
	}
	return out, nil
}

func PlainQuarticPackageScores(features [][]float64, landmarks int) []float64 {
	out := make([]float64, len(features))
	for r := 0; r < landmarks; r++ {
		ref := deterministicUnitReference(r, len(features[0]))
		alpha := math.Sin(0.013*float64(r+1)) / math.Sqrt(float64(landmarks))
		for i, x := range features {
			var dot, n2 float64
			for d, v := range x {
				dot += v * ref[d]
				n2 += v * v
			}
			u := 2.0*dot - n2 - 1.0
			u2 := u * u
			p := ExpPoly4[0] + ExpPoly4[1]*u + ExpPoly4[2]*u2 + ExpPoly4[3]*u*u2 + ExpPoly4[4]*u2*u2
			out[i] += alpha * p
		}
	}
	return out
}
