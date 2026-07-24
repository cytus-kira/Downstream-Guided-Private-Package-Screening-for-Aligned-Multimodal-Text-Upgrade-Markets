#include <seal/seal.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

using namespace std;
using namespace seal;

// ============ 工具：随机矩阵 ============
static vector<vector<double>> random_matrix(size_t m, size_t n, double lo=-1.0, double hi=1.0) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(lo, hi);
    vector<vector<double>> M(m, vector<double>(n));
    for (size_t i=0;i<m;++i) for (size_t j=0;j<n;++j) M[i][j]=dist(rng);
    return M;
}

// ============ Debug：打印 ct 状态 ============
static inline void print_ct_info(const string &tag, const Ciphertext &ct, const SEALContext &ctx) {
    auto cd = ctx.get_context_data(ct.parms_id());
    cout << "   [ct] " << tag
         << " | scale=" << ct.scale()
         << " | size=" << ct.size()
         << " | chain_index=" << (cd ? cd->chain_index() : -1)
         << "\n";
}

static inline void require_chain_index(const string &tag, const Ciphertext &ct,
                                       const SEALContext &ctx, size_t expected) {
    auto cd = ctx.get_context_data(ct.parms_id());
    if (!cd || cd->chain_index() != expected) {
        throw runtime_error(tag + " has unexpected CKKS chain index.");
    }
}

static inline void decrypt_decode_check(const string &tag, const Ciphertext &ct,
                                        Decryptor &decryptor, CKKSEncoder &encoder) {
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    vector<double> decoded;
    encoder.decode(pt, decoded);
    if (decoded.empty() || !std::isfinite(decoded.front())) {
        throw runtime_error(tag + " failed CKKS decrypt/decode validation.");
    }
    cout << "   [dec] " << tag << " | first_slot=" << decoded.front() << "\n";
}

// ============ 对齐：cipher-cipher ============
static inline void match_level_and_scale_cipher(Ciphertext &a, Ciphertext &b,
                                                Evaluator &evaluator, const SEALContext &context) {
    auto ac = context.get_context_data(a.parms_id());
    auto bc = context.get_context_data(b.parms_id());
    if (!ac || !bc) throw runtime_error("Invalid parms_id in match_level_and_scale_cipher.");

    if (ac->chain_index() < bc->chain_index()) {
        evaluator.mod_switch_to_inplace(b, a.parms_id());
    } else if (ac->chain_index() > bc->chain_index()) {
        evaluator.mod_switch_to_inplace(a, b.parms_id());
    }
    // 强制对齐 scale（你现有代码风格）
    b.scale() = a.scale();
}

// ============ 对齐：cipher-plain ============
static inline void match_level_and_scale(Ciphertext &a, Plaintext &b,
                                         Evaluator &evaluator, const SEALContext &context) {
    auto ac = context.get_context_data(a.parms_id());
    auto bc = context.get_context_data(b.parms_id());
    if (!ac || !bc) throw runtime_error("Invalid parms_id in match_level_and_scale.");

    if (ac->chain_index() < bc->chain_index()) {
        evaluator.mod_switch_to_inplace(b, a.parms_id());
    } else if (ac->chain_index() > bc->chain_index()) {
        evaluator.mod_switch_to_inplace(a, b.parms_id());
    }
    b.scale() = a.scale();
}

// Raise a CKKS scale without consuming a modulus level by multiplying by
// an encoding of 1.0 at scale target/current. This preserves the decoded value.
static inline void lift_scale_inplace(Ciphertext &ct, double target_scale,
                                      CKKSEncoder &encoder, Evaluator &evaluator) {
    double ratio = target_scale / ct.scale();
    if (std::abs(std::log2(ratio)) < 1e-9) {
        ct.scale() = target_scale;
        return;
    }
    if (ratio < 1.0) {
        throw runtime_error("lift_scale_inplace only supports increasing the scale.");
    }

    Plaintext pt_one;
    encoder.encode(1.0, ratio, pt_one);
    evaluator.mod_switch_to_inplace(pt_one, ct.parms_id());
    evaluator.multiply_plain_inplace(ct, pt_one);
    ct.scale() = target_scale;
}

// ============ encode col-major (m x k) into one ct ============
static Ciphertext encode_matrix_colmajor(const vector<vector<double>> &A,
                                         CKKSEncoder &encoder, Encryptor &encryptor, double scale) {
    size_t m=A.size(), k=A[0].size();
    vector<double> flat(m*k, 0.0);
    for (size_t col=0; col<k; ++col)
        for (size_t row=0; row<m; ++row)
            flat[col*m + row] = A[row][col];

    Plaintext pt;
    encoder.encode(flat, scale, pt);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

// ============ SIMD matmul: A(ct, m x k) * B(pt, k x n) -> C(ct, m x n) ============
static Ciphertext simd_encrypted_matmul(const Ciphertext &encrypted_A,
                                        size_t m, size_t k, size_t n,
                                        const vector<vector<double>> &B,
                                        CKKSEncoder &encoder,
                                        Evaluator &evaluator,
                                        GaloisKeys &gal_keys) {
    size_t slot_count = encoder.slot_count();
    double scale = encrypted_A.scale();
    parms_id_type base_parms = encrypted_A.parms_id();

    // k should be power-of-two in your implementation assumption
    size_t logk = static_cast<size_t>(std::log2((double)k));

    Ciphertext acc_all;
    bool first=true;

    vector<double> slot_mask(slot_count, 0.0);
    for (size_t row=0; row<m; ++row) slot_mask[row]=1.0;
    Plaintext slot_mask_plain;

    for (size_t j=0; j<n; ++j) {
        vector<double> mask(slot_count, 0.0);
        for (size_t col=0; col<k; ++col)
            for (size_t row=0; row<m; ++row)
                mask[col*m + row] = B[col][j];

        Plaintext mask_plain;
        encoder.encode(mask, scale, mask_plain);
        evaluator.mod_switch_to_inplace(mask_plain, base_parms);

        Ciphertext AB;
        evaluator.multiply_plain(encrypted_A, mask_plain, AB);
        evaluator.rescale_to_next_inplace(AB);

        for (size_t l=0; l<logk; ++l) {
            size_t step=(1ULL<<l)*m;
            Ciphertext rot;
            evaluator.rotate_vector(AB, static_cast<int>(step), gal_keys, rot);
            evaluator.add_inplace(AB, rot);
        }

        encoder.encode(slot_mask, AB.scale(), slot_mask_plain);
        evaluator.mod_switch_to_inplace(slot_mask_plain, AB.parms_id());
        slot_mask_plain.scale() = AB.scale();

        Ciphertext masked;
        evaluator.multiply_plain(AB, slot_mask_plain, masked);
        evaluator.rescale_to_next_inplace(masked);

        Ciphertext shifted;
        if (j>0) evaluator.rotate_vector(masked, -static_cast<int>(j*m), gal_keys, shifted);
        else shifted = masked;

        if (first) { acc_all = shifted; first=false; }
        else {
            evaluator.mod_switch_to_inplace(shifted, acc_all.parms_id());
            shifted.scale() = acc_all.scale();
            evaluator.add_inplace(acc_all, shifted);
        }
    }
    return acc_all;
}

// ============ GELU forward poly (your style): gelu(x) ≈ a x^2 + b x + c ============
static Ciphertext gelu_poly_approx(Ciphertext &x,
                                   CKKSEncoder &encoder,
                                   Evaluator &evaluator,
                                   RelinKeys &relin_keys,
                                   const SEALContext &context) {
    // x^2
    Ciphertext x2;
    evaluator.square(x, x2);
    evaluator.relinearize_inplace(x2, relin_keys);
    evaluator.rescale_to_next_inplace(x2);

    double a = 0.250371604909945;
    double b = 0.4999999999902377;
    double c = 0.05076363910865732;

    double sc = x.scale();
    Plaintext pt_a, pt_b, pt_c;
    encoder.encode(a, sc, pt_a);
    encoder.encode(b, sc, pt_b);
    encoder.encode(c, sc, pt_c);

    // ax2
    Ciphertext ax2;
    Plaintext pt_a2 = pt_a;
    match_level_and_scale(x2, pt_a2, evaluator, context);
    evaluator.multiply_plain(x2, pt_a2, ax2);
    evaluator.rescale_to_next_inplace(ax2);

    // bx
    Ciphertext bx;
    Plaintext pt_b2 = pt_b;
    match_level_and_scale(x, pt_b2, evaluator, context);
    evaluator.multiply_plain(x, pt_b2, bx);
    evaluator.rescale_to_next_inplace(bx);

    match_level_and_scale_cipher(ax2, bx, evaluator, context);
    evaluator.add_inplace(ax2, bx);

    Plaintext pt_c2 = pt_c;
    evaluator.mod_switch_to_inplace(pt_c2, ax2.parms_id());
    pt_c2.scale() = ax2.scale();
    evaluator.add_plain_inplace(ax2, pt_c2);

    return ax2;
}

// ============ GELU' (consistent with forward poly): gelu'(x)=2a x + b ============
static Ciphertext gelu_poly_deriv(Ciphertext &x,
                                  CKKSEncoder &encoder,
                                  Evaluator &evaluator,
                                  const SEALContext &context) {
    double a = 0.250371604909945;
    double b = 0.4999999999902377;
    double two_a = 2.0 * a;

    double sc = x.scale();
    Plaintext pt_two_a, pt_b;
    encoder.encode(two_a, sc, pt_two_a);
    encoder.encode(b, sc, pt_b);

    match_level_and_scale(x, pt_two_a, evaluator, context);
    Ciphertext out;
    evaluator.multiply_plain(x, pt_two_a, out);
    evaluator.rescale_to_next_inplace(out);

    evaluator.mod_switch_to_inplace(pt_b, out.parms_id());
    pt_b.scale() = out.scale();
    evaluator.add_plain_inplace(out, pt_b);

    return out;
}

// ============ sigmoid poly approx (degree-3) ============
static Ciphertext sigmoid_poly3(Ciphertext &x,
                                CKKSEncoder &encoder,
                                Evaluator &evaluator,
                                RelinKeys &relin_keys,
                                const SEALContext &context) {
    // A simple cubic approximation on a moderate range.
    // sigma(x) ≈ 0.5 + 0.15012 x - 0.001593 x^3
    // (You can replace coefficients with your fitted Chebyshev if needed.)
    const double c0 = 0.5;
    const double c1 = 0.15012;
    const double c3 = -0.001593;

    // x^2
    Ciphertext x2;
    evaluator.square(x, x2);
    evaluator.relinearize_inplace(x2, relin_keys);
    evaluator.rescale_to_next_inplace(x2);

    // x^3 = x^2 * x
    Ciphertext x_mod = x;
    evaluator.mod_switch_to_inplace(x_mod, x2.parms_id());
    x_mod.scale() = x2.scale();

    Ciphertext x3;
    evaluator.multiply(x2, x_mod, x3);
    evaluator.relinearize_inplace(x3, relin_keys);
    evaluator.rescale_to_next_inplace(x3);

    // c1*x
    Plaintext pt_c1;
    encoder.encode(c1, x.scale(), pt_c1);
    match_level_and_scale(x, pt_c1, evaluator, context);
    Ciphertext t1;
    evaluator.multiply_plain(x, pt_c1, t1);
    evaluator.rescale_to_next_inplace(t1);

    // c3*x^3
    Plaintext pt_c3;
    encoder.encode(c3, x3.scale(), pt_c3);
    evaluator.mod_switch_to_inplace(pt_c3, x3.parms_id());
    pt_c3.scale() = x3.scale();

    Ciphertext t3;
    evaluator.multiply_plain(x3, pt_c3, t3);
    evaluator.rescale_to_next_inplace(t3);

    // align & add
    match_level_and_scale_cipher(t3, t1, evaluator, context);
    evaluator.add_inplace(t3, t1);

    // + c0
    Plaintext pt_c0;
    encoder.encode(c0, t3.scale(), pt_c0);
    evaluator.mod_switch_to_inplace(pt_c0, t3.parms_id());
    pt_c0.scale() = t3.scale();
    evaluator.add_plain_inplace(t3, pt_c0);

    return t3;
}

// ============ BCE-with-logits grad: d = (sigmoid(logit)-y)/B ============
static Ciphertext bce_with_logits_grad_poly(Ciphertext &ct_logits,  // [B,1] in first block
                                            const vector<double> &y_vec, // length B (plaintext labels)
                                            size_t B,
                                            CKKSEncoder &encoder,
                                            Evaluator &evaluator,
                                            RelinKeys &relin_keys,
                                            GaloisKeys &gal_keys,
                                            const SEALContext &context,
                                            double target_scale) {
    Ciphertext sig = sigmoid_poly3(ct_logits, encoder, evaluator, relin_keys,
                                   context);

    // subtract y (only first B slots are meaningful)
    size_t slot_count = encoder.slot_count();
    vector<double> y_slots(slot_count, 0.0);
    for (size_t i=0;i<B;++i) y_slots[i] = y_vec[i];

    Plaintext pt_y;
    encoder.encode(y_slots, sig.scale(), pt_y);
    evaluator.mod_switch_to_inplace(pt_y, sig.parms_id());
    pt_y.scale() = sig.scale();

    Ciphertext diff;
    evaluator.sub_plain(sig, pt_y, diff);

    // multiply by 1/B
    double invB = 1.0 / (double)B;
    Plaintext pt_invB;
    encoder.encode(invB, diff.scale(), pt_invB);
    evaluator.mod_switch_to_inplace(pt_invB, diff.parms_id());
    pt_invB.scale() = diff.scale();

    evaluator.multiply_plain_inplace(diff, pt_invB);
    evaluator.rescale_to_next_inplace(diff);

    diff.scale() = target_scale;
    return diff; // [B,1] in first block
}

// ============ Broadcast: replicate first block (size B) into d_blocks blocks by doubling rotations ============
static Ciphertext broadcast_first_block_pow2(const Ciphertext &ct_vec_first_block,
                                             size_t B,
                                             size_t blocks_pow2,
                                             Evaluator &evaluator,
                                             GaloisKeys &gal_keys,
                                             const SEALContext &context) {
    // blocks_pow2 must be power of two
    Ciphertext acc = ct_vec_first_block;
    size_t blocks=1;
    while (blocks < blocks_pow2) {
        size_t shift = blocks * B; // shift by blocks*B slots
        Ciphertext rot;
        evaluator.rotate_vector(acc, static_cast<int>(shift), gal_keys, rot);
        evaluator.add_inplace(acc, rot);
        blocks *= 2;
    }
    return acc;
}

// ============ Concat (half/half mask) WITHOUT rescale (so backward can chain) ============
static Ciphertext concat_half_mask_no_rescale(const Ciphertext &ct1, const Ciphertext &ct2,
                                              CKKSEncoder &encoder,
                                              Evaluator &evaluator,
                                              GaloisKeys &gal_keys,
                                              const SEALContext &context,
                                              double target_scale) {
    size_t slot_count = encoder.slot_count();
    size_t half = slot_count / 2;

    // mask1: first half 1, second half 0
    vector<double> mask1(slot_count, 0.0);
    for (size_t i=0;i<half;++i) mask1[i]=1.0;

    // encode mask with scale=1.0 to avoid scale blow-up => no rescale needed
    Plaintext pt_mask;
    encoder.encode(mask1, 1.0, pt_mask);
    evaluator.mod_switch_to_inplace(pt_mask, ct1.parms_id());
    // no need to force same scale; multiply_plain will multiply scale by pt.scale(=1)

    Ciphertext ct1m, ct2m;
    evaluator.multiply_plain(ct1, pt_mask, ct1m);
    evaluator.multiply_plain(ct2, pt_mask, ct2m);

    // rotate ct2m into second half
    Ciphertext ct2r;
    evaluator.rotate_vector(ct2m, static_cast<int>(half), gal_keys, ct2r);

    // align level/scale for add
    Ciphertext a = ct1m, b = ct2r;
    match_level_and_scale_cipher(a, b, evaluator, context);
    evaluator.add_inplace(a, b);

    a.scale() = target_scale;
    return a;
}

// ============ Backward of concat_half_mask_no_rescale ============
static void concat_half_backward_no_rescale(const Ciphertext &d_merged,
                                            CKKSEncoder &encoder,
                                            Evaluator &evaluator,
                                            GaloisKeys &gal_keys,
                                            Ciphertext &d_ct1, Ciphertext &d_ct2) {
    size_t half = encoder.slot_count() / 2;
    // forward: merged = ct1_masked + rot(ct2_masked, +half)
    // backward:
    // d_ct1_masked += d_merged
    // d_ct2_masked += rot(d_merged, -half)
    d_ct1 = d_merged;

    Ciphertext rot_back;
    evaluator.rotate_vector(d_merged, -static_cast<int>(half), gal_keys, rot_back);
    d_ct2 = rot_back;
}

// ============ BOPA core forward cache ============
struct BopaCache {
    size_t m{0}, k{0};
    Ciphertext A, B, C;       // inputs (ct)
    Ciphertext prod_ab;       // after A⊙B (+rescale)
    Ciphertext sum_ab;        // after sumcols (no rescale)
    Ciphertext out;           // after sum_ab ⊙ C (+rescale)
};

static size_t ilog2_exact(size_t x) {
    size_t r=0;
    while ((size_t(1)<<r) < x) r++;
    return r;
}

// sumcols forward (k must be power-of-two): sum over column blocks by +step rotations
static Ciphertext sumcols_forward(const Ciphertext &ct_mk, size_t m, size_t k,
                                  Evaluator &evaluator, GaloisKeys &gal_keys) {
    Ciphertext acc = ct_mk;
    size_t L = ilog2_exact(k);
    for (size_t l=0;l<L;++l) {
        size_t step=(size_t(1)<<l)*m;
        Ciphertext rot;
        evaluator.rotate_vector(acc, static_cast<int>(step), gal_keys, rot);
        evaluator.add_inplace(acc, rot);
    }
    return acc;
}

// sumcols backward (adjoint): sum over inverse rotations
static Ciphertext sumcols_backward(const Ciphertext &d_sum, size_t m, size_t k,
                                   Evaluator &evaluator, GaloisKeys &gal_keys) {
    Ciphertext acc = d_sum;
    size_t L = ilog2_exact(k);
    for (size_t l=0;l<L;++l) {
        size_t step=(size_t(1)<<l)*m;
        Ciphertext rot;
        evaluator.rotate_vector(acc, -static_cast<int>(step), gal_keys, rot);
        evaluator.add_inplace(acc, rot);
    }
    return acc;
}

// BOPA forward: out = (sumcols(A⊙B)) ⊙ C
static BopaCache bopa_forward(const Ciphertext &ctA, const Ciphertext &ctB, const Ciphertext &ctC,
                              size_t m, size_t k,
                              Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys,
                              const SEALContext &context, double target_scale) {
    BopaCache cache;
    cache.m=m; cache.k=k;
    cache.A=ctA; cache.B=ctB; cache.C=ctC;

    // prod_ab = A⊙B
    Ciphertext prod;
    Ciphertext A_=ctA, B_=ctB;
    match_level_and_scale_cipher(A_, B_, evaluator, context);
    evaluator.multiply(A_, B_, prod);
    evaluator.relinearize_inplace(prod, relin_keys);
    evaluator.rescale_to_next_inplace(prod);
    prod.scale() = target_scale;
    cache.prod_ab = prod;

    // sumcols
    cache.sum_ab = sumcols_forward(cache.prod_ab, m, k, evaluator, gal_keys);

    // out = sum_ab ⊙ C
    Ciphertext C_ = ctC;
    evaluator.mod_switch_to_inplace(C_, cache.sum_ab.parms_id());
    C_.scale() = cache.sum_ab.scale();

    Ciphertext out;
    Ciphertext S_=cache.sum_ab, C2_=C_;
    match_level_and_scale_cipher(S_, C2_, evaluator, context);
    evaluator.multiply(S_, C2_, out);
    evaluator.relinearize_inplace(out, relin_keys);
    evaluator.rescale_to_next_inplace(out);
    out.scale() = target_scale;
    cache.out = out;

    return cache;
}

// BOPA backward: given d_out, compute dA,dB,dC (all ciphertext)
struct BopaGrads {
    Ciphertext dA, dB, dC;
};

static BopaGrads bopa_backward(const BopaCache &fwd,
                               const Ciphertext &d_out,
                               CKKSEncoder &encoder, Evaluator &evaluator,
                               GaloisKeys &gal_keys, RelinKeys &relin_keys,
                               const SEALContext &context, double target_scale) {
    BopaGrads g;
    size_t m=fwd.m, k=fwd.k;

    // d_sum = d_out ⊙ C
    Ciphertext C_ = fwd.C;
    evaluator.mod_switch_to_inplace(C_, d_out.parms_id());
    lift_scale_inplace(C_, target_scale, encoder, evaluator);

    Ciphertext dout_ = d_out, C2_=C_;
    match_level_and_scale_cipher(dout_, C2_, evaluator, context);

    Ciphertext d_sum;
    evaluator.multiply(dout_, C2_, d_sum);
    evaluator.relinearize_inplace(d_sum, relin_keys);
    evaluator.rescale_to_next_inplace(d_sum);
    d_sum.scale() = target_scale;

    // dC = d_out ⊙ sum_ab
    Ciphertext S_ = fwd.sum_ab;
    evaluator.mod_switch_to_inplace(S_, d_out.parms_id());
    lift_scale_inplace(S_, target_scale, encoder, evaluator);

    Ciphertext dC;
    Ciphertext dout2_=d_out, S2_=S_;
    match_level_and_scale_cipher(dout2_, S2_, evaluator, context);
    evaluator.multiply(dout2_, S2_, dC);
    evaluator.relinearize_inplace(dC, relin_keys);
    evaluator.rescale_to_next_inplace(dC);
    dC.scale() = target_scale;
    g.dC = dC;

    // d_prod = sumcols_backward(d_sum)
    Ciphertext d_prod = sumcols_backward(d_sum, m, k, evaluator, gal_keys);

    // dA = d_prod ⊙ B
    Ciphertext B_ = fwd.B;
    evaluator.mod_switch_to_inplace(B_, d_prod.parms_id());
    lift_scale_inplace(B_, target_scale, encoder, evaluator);

    Ciphertext dA;
    Ciphertext dp_=d_prod, B3_=B_;
    match_level_and_scale_cipher(dp_, B3_, evaluator, context);
    evaluator.multiply(dp_, B3_, dA);
    evaluator.relinearize_inplace(dA, relin_keys);
    evaluator.rescale_to_next_inplace(dA);
    dA.scale() = target_scale;
    g.dA = dA;

    // dB = d_prod ⊙ A
    Ciphertext A_ = fwd.A;
    evaluator.mod_switch_to_inplace(A_, d_prod.parms_id());
    lift_scale_inplace(A_, target_scale, encoder, evaluator);

    Ciphertext dB;
    Ciphertext dp2_=d_prod, A3_=A_;
    match_level_and_scale_cipher(dp2_, A3_, evaluator, context);
    evaluator.multiply(dp2_, A3_, dB);
    evaluator.relinearize_inplace(dB, relin_keys);
    evaluator.rescale_to_next_inplace(dB);
    dB.scale() = target_scale;
    g.dB = dB;

    return g;
}

// ============ 网络 forward cache ============
struct NetCache {
    size_t B{0};          // rows
    size_t k_in{0};       // merged input width (e.g., 2048)
    size_t h{0};          // hidden (e.g., 1024)

    // BOPA two branches
    BopaCache bopa1;
    BopaCache bopa2;

    // concat / square / fc / gelu
    Ciphertext merged;        // concat result
    Ciphertext merged_sq;     // merged^2
    Ciphertext fc1;           // FC1 output [B,h]
    Ciphertext gelu1;         // GELU(fc1)
    Ciphertext logits;        // FC2 output [B,1] (first block)

    // plaintext weights used in FC
    vector<vector<double>> W_fc1;    // [k_in, h]
    vector<vector<double>> W_fc2;    // [h, 1]
};

// ============ forward: BOPA×2 -> concat -> square -> FC1 -> GELU -> FC2 ============
static NetCache net_forward_demo(SEALContext &context,
                                 CKKSEncoder &encoder,
                                 Encryptor &encryptor,
                                 Evaluator &evaluator,
                                 GaloisKeys &gal_keys,
                                 RelinKeys &relin_keys,
                                 double scale,
                                 double target_scale) {
    NetCache cache;

    // The small audit build preserves the same operation depth while reducing
    // rotation counts for parameter-chain validation.
#ifdef SGF_AUDIT_SMALL
    size_t B = 2;
    size_t k = 2;
    size_t k_in = 4;
    size_t h = 2;
#else
    size_t B = 32;
    size_t k = 256;    // for bopa inner width
    size_t k_in = 512; // concat gives 2 branches -> roughly doubles; we demo as 2048
    size_t h = 256;
#endif

    cache.B=B; cache.k_in=k_in; cache.h=h;

    // BOPA inputs (ct) - here we just encrypt random matrices
    auto A1p = random_matrix(B, k, -1.0, 1.0);
    auto B1p = random_matrix(B, k, -1.0, 1.0);
    auto C1p = random_matrix(B, k, -1.0, 1.0);

    auto A2p = random_matrix(B, k, -1.0, 1.0);
    auto B2p = random_matrix(B, k, -1.0, 1.0);
    auto C2p = random_matrix(B, k, -1.0, 1.0);

    Ciphertext ctA1 = encode_matrix_colmajor(A1p, encoder, encryptor, scale);
    Ciphertext ctB1 = encode_matrix_colmajor(B1p, encoder, encryptor, scale);
    Ciphertext ctC1 = encode_matrix_colmajor(C1p, encoder, encryptor, scale);

    Ciphertext ctA2 = encode_matrix_colmajor(A2p, encoder, encryptor, scale);
    Ciphertext ctB2 = encode_matrix_colmajor(B2p, encoder, encryptor, scale);
    Ciphertext ctC2 = encode_matrix_colmajor(C2p, encoder, encryptor, scale);

    // BOPA forward
    cache.bopa1 = bopa_forward(ctA1, ctB1, ctC1, B, k, evaluator, gal_keys, relin_keys, context, target_scale);
    cache.bopa2 = bopa_forward(ctA2, ctB2, ctC2, B, k, evaluator, gal_keys, relin_keys, context, target_scale);

    // concat (no-rescale version to keep chainable backward)
    cache.merged = concat_half_mask_no_rescale(cache.bopa1.out, cache.bopa2.out,
                                              encoder, evaluator, gal_keys, context, target_scale);

    // square
    evaluator.square(cache.merged, cache.merged_sq);
    evaluator.relinearize_inplace(cache.merged_sq, relin_keys);
    evaluator.rescale_to_next_inplace(cache.merged_sq);
    cache.merged_sq.scale() = target_scale;

    // FC1 weights: [k_in, h]
    cache.W_fc1 = random_matrix(k_in, h, -0.5, 0.5);

    // FC1: merged_sq [B,k_in] * W_fc1 [k_in,h]
    cache.fc1 = simd_encrypted_matmul(cache.merged_sq, B, k_in, h, cache.W_fc1, encoder, evaluator, gal_keys);

    // GELU
    Ciphertext fc1_copy = cache.fc1;
    cache.gelu1 = gelu_poly_approx(fc1_copy, encoder, evaluator, relin_keys, context);
    cache.gelu1.scale() = target_scale;

    // FC2 weights: [h, 1]
    cache.W_fc2 = random_matrix(h, 1, -0.5, 0.5);

    // FC2: gelu1 [B,h] * W_fc2 [h,1]
    cache.logits = simd_encrypted_matmul(cache.gelu1, B, h, 1, cache.W_fc2, encoder, evaluator, gal_keys);

    return cache;
}

// ============ backward: logits -> gelu1 -> fc1 -> merged_sq -> merged -> bopa1/bopa2 -> (A/B/C grads) ============
struct NetGrads {
    Ciphertext d_logits;
    Ciphertext d_gelu1;
    Ciphertext d_fc1;
    Ciphertext d_merged_sq;
    Ciphertext d_merged;

    Ciphertext d_bopa1_out;
    Ciphertext d_bopa2_out;

    BopaGrads g1;
    BopaGrads g2;
};

static NetGrads net_backward_demo(const NetCache &cache,
                                  const vector<double> &y_vec, // length B
                                  SEALContext &context,
                                  CKKSEncoder &encoder,
                                  Evaluator &evaluator,
                                  GaloisKeys &gal_keys,
                                  RelinKeys &relin_keys,
                                  double target_scale) {
    NetGrads g;
    size_t B=cache.B, h=cache.h, k_in=cache.k_in;

    // 1) d_logits = (sigmoid(logits)-y)/B   (polynomial sigmoid)
    Ciphertext logits_copy = cache.logits;
    lift_scale_inplace(logits_copy, target_scale, encoder, evaluator);
    g.d_logits = bce_with_logits_grad_poly(logits_copy, y_vec, B,
                                          encoder, evaluator, relin_keys, gal_keys, context, target_scale);
    print_ct_info("d_logits", g.d_logits, context);

    // 2) d_gelu1 = d_logits @ W_fc2^T
    //    W_fc2 is [h,1] => W_fc2^T is [1,h]
    //    We do: broadcast d_logits to [B,h] then block-wise scale by W_fc2^T
    //    broadcast blocks must be power-of-two => h is 1024 in demo
    Ciphertext d_logits_broad = broadcast_first_block_pow2(g.d_logits, B, h, evaluator, gal_keys, context);
    d_logits_broad.scale() = g.d_logits.scale();

    // build plaintext weight mask (repeat each weight on its block)
    size_t slot_count = encoder.slot_count();
    if (B*h > slot_count) throw runtime_error("B*h exceeds slot_count in backward.");

    vector<double> wmask(slot_count, 0.0);
    for (size_t col=0; col<h; ++col) {
        double w = cache.W_fc2[col][0];
        size_t base = col * B;
        for (size_t i=0;i<B;++i) wmask[base+i] = w;
    }

    Plaintext pt_w;
    encoder.encode(wmask, d_logits_broad.scale(), pt_w);
    evaluator.mod_switch_to_inplace(pt_w, d_logits_broad.parms_id());
    pt_w.scale() = d_logits_broad.scale();

    evaluator.multiply_plain(d_logits_broad, pt_w, g.d_gelu1);
    evaluator.rescale_to_next_inplace(g.d_gelu1);
    g.d_gelu1.scale() = target_scale;
    print_ct_info("d_gelu1", g.d_gelu1, context);

    // 3) d_fc1 = d_gelu1 ⊙ gelu'(fc1)
    Ciphertext fc1_copy = cache.fc1;
    Ciphertext gelu_p = gelu_poly_deriv(fc1_copy, encoder, evaluator, context);
    lift_scale_inplace(gelu_p, target_scale, encoder, evaluator);

    match_level_and_scale_cipher(g.d_gelu1, gelu_p, evaluator, context);
    evaluator.multiply(g.d_gelu1, gelu_p, g.d_fc1);
    evaluator.relinearize_inplace(g.d_fc1, relin_keys);
    evaluator.rescale_to_next_inplace(g.d_fc1);
    g.d_fc1.scale() = target_scale;
    print_ct_info("d_fc1", g.d_fc1, context);

    // 4) d_merged_sq = d_fc1 @ W_fc1^T
    //    forward: merged_sq [B,k_in] * W_fc1 [k_in,h] = fc1 [B,h]
    //    backward to input: d_merged_sq = d_fc1 [B,h] * (W_fc1)^T [h,k_in]
    vector<vector<double>> W_fc1_T(h, vector<double>(k_in, 0.0));
    for (size_t i=0;i<k_in;++i)
        for (size_t j=0;j<h;++j)
            W_fc1_T[j][i] = cache.W_fc1[i][j];

    g.d_merged_sq = simd_encrypted_matmul(g.d_fc1, B, h, k_in, W_fc1_T, encoder, evaluator, gal_keys);
    g.d_merged_sq.scale() = target_scale;
    print_ct_info("d_merged_sq", g.d_merged_sq, context);

    // 5) square backward: merged_sq = merged^2 => d_merged = 2*merged ⊙ d_merged_sq
    Ciphertext merged_copy = cache.merged;
    lift_scale_inplace(merged_copy, target_scale, encoder, evaluator);
    match_level_and_scale_cipher(merged_copy, g.d_merged_sq, evaluator, context);

    Ciphertext tmp;
    evaluator.multiply(merged_copy, g.d_merged_sq, tmp);
    evaluator.relinearize_inplace(tmp, relin_keys);
    evaluator.rescale_to_next_inplace(tmp);
    tmp.scale() = target_scale;

    Plaintext pt2;
    encoder.encode(2.0, tmp.scale(), pt2);
    evaluator.mod_switch_to_inplace(pt2, tmp.parms_id());
    pt2.scale() = tmp.scale();

    evaluator.multiply_plain(tmp, pt2, g.d_merged);
    evaluator.rescale_to_next_inplace(g.d_merged);
    g.d_merged.scale() = target_scale;
    print_ct_info("d_merged", g.d_merged, context);

    // 6) concat backward: split d_merged into two branch grads
    concat_half_backward_no_rescale(g.d_merged, encoder, evaluator, gal_keys, g.d_bopa1_out, g.d_bopa2_out);
    g.d_bopa1_out.scale() = target_scale;
    g.d_bopa2_out.scale() = target_scale;
    print_ct_info("d_bopa1_out", g.d_bopa1_out, context);
    print_ct_info("d_bopa2_out", g.d_bopa2_out, context);

    // 7) bopa backward into (A,B,C) for each branch
    g.g1 = bopa_backward(cache.bopa1, g.d_bopa1_out, encoder, evaluator,
                         gal_keys, relin_keys, context, target_scale);
    g.g2 = bopa_backward(cache.bopa2, g.d_bopa2_out, encoder, evaluator,
                         gal_keys, relin_keys, context, target_scale);

    print_ct_info("g1.dA", g.g1.dA, context);
    print_ct_info("g1.dB", g.g1.dB, context);
    print_ct_info("g1.dC", g.g1.dC, context);

    print_ct_info("g2.dA", g.g2.dA, context);
    print_ct_info("g2.dB", g.g2.dB, context);
    print_ct_info("g2.dC", g.g2.dC, context);

    return g;
}

int main() {
    cout << unitbuf;
    // 23 primes, 840 total bits: 50-bit terminal data prime, twelve
    // backward 40-bit primes, nine forward 30-bit primes, and a 40-bit
    // key-switching special prime. This stays below the N=32768 tc128
    // maximum of 881 bits.
    size_t poly_modulus_degree = 16384 * 2;
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
        50,
        40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
        30, 30, 30, 30, 30, 30, 30, 30, 30,
        40
    }));

    SEALContext context(parms);
    if (!context.parameters_set()) {
        throw runtime_error(string("Invalid CKKS parameters: ") +
                            context.parameter_error_message());
    }
    KeyGenerator keygen(context);

    PublicKey pk; keygen.create_public_key(pk);
    SecretKey sk = keygen.secret_key();

    GaloisKeys gk; keygen.create_galois_keys(gk);
    RelinKeys rk; keygen.create_relin_keys(rk);

    Encryptor encryptor(context, pk);
    Decryptor decryptor(context, sk);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    double forward_scale = pow(2.0, 30);
    double backward_scale = pow(2.0, 40);

    cout << "[init] slot_count=" << encoder.slot_count()
         << " | total_coeff_modulus_bits="
         << context.key_context_data()->total_coeff_modulus_bit_count()
         << " | initial_chain_index="
         << context.first_context_data()->chain_index() << "\n";

    // ---- forward ----
    NetCache cache = net_forward_demo(context, encoder, encryptor, evaluator, gk, rk,
                                      forward_scale, forward_scale);
    cout << "[forward] states:\n";
    print_ct_info("bopa1.out", cache.bopa1.out, context);
    print_ct_info("bopa2.out", cache.bopa2.out, context);
    print_ct_info("merged", cache.merged, context);
    print_ct_info("merged_sq", cache.merged_sq, context);
    print_ct_info("fc1", cache.fc1, context);
    print_ct_info("gelu1", cache.gelu1, context);
    print_ct_info("logits", cache.logits, context);
    require_chain_index("bopa1.out", cache.bopa1.out, context, 19);
    require_chain_index("bopa2.out", cache.bopa2.out, context, 19);
    require_chain_index("merged", cache.merged, context, 19);
    require_chain_index("merged_sq", cache.merged_sq, context, 18);
    require_chain_index("fc1", cache.fc1, context, 16);
    require_chain_index("gelu1", cache.gelu1, context, 14);
    require_chain_index("logits", cache.logits, context, 12);

    // ---- labels (plaintext) ----
    vector<double> y(cache.B, 0.0);
    for (size_t i=0;i<cache.B;++i) y[i] = (i%2);

    // ---- backward ----
    cout << "\n[backward] running...\n";
    NetGrads grads = net_backward_demo(cache, y, context, encoder, evaluator, gk, rk,
                                       backward_scale);
    require_chain_index("d_logits", grads.d_logits, context, 8);
    require_chain_index("d_gelu1", grads.d_gelu1, context, 7);
    require_chain_index("d_fc1", grads.d_fc1, context, 6);
    require_chain_index("d_merged_sq", grads.d_merged_sq, context, 4);
    require_chain_index("d_merged", grads.d_merged, context, 2);
    require_chain_index("g1.dA", grads.g1.dA, context, 0);
    require_chain_index("g1.dB", grads.g1.dB, context, 0);
    require_chain_index("g1.dC", grads.g1.dC, context, 1);
    require_chain_index("g2.dA", grads.g2.dA, context, 0);
    require_chain_index("g2.dB", grads.g2.dB, context, 0);
    require_chain_index("g2.dC", grads.g2.dC, context, 1);
    decrypt_decode_check("logits", cache.logits, decryptor, encoder);
    decrypt_decode_check("g1.dA", grads.g1.dA, decryptor, encoder);
    decrypt_decode_check("g1.dB", grads.g1.dB, decryptor, encoder);
    decrypt_decode_check("g2.dA", grads.g2.dA, decryptor, encoder);
    decrypt_decode_check("g2.dB", grads.g2.dB, decryptor, encoder);

    cout << "\n[done]\n";
    return 0;
}
