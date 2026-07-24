#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace std;
using namespace seal;

struct Args {
    size_t poly_modulus_degree = 32768;
    vector<int> coeff_mod_bits = {60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60};
    double scale_bits = 50.0;
    vector<size_t> dims = {128, 256, 512, 1024};
    size_t repeats = 10;
    size_t warmups = 2;
    size_t typiclust_centroids = 32;
    string output_csv = "seal_cipher_compute_benchmark.csv";
    bool verbose = true;
};

struct TimerSummary {
    double mean_ms{0.0};
    double std_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

static vector<string> split_csv(const string &s) {
    vector<string> out;
    string cur;
    stringstream ss(s);
    while (getline(ss, cur, ',')) {
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

static vector<int> parse_int_csv(const string &s) {
    vector<int> out;
    for (const auto &x : split_csv(s)) out.push_back(stoi(x));
    return out;
}

static vector<size_t> parse_size_t_csv(const string &s) {
    vector<size_t> out;
    for (const auto &x : split_csv(s)) out.push_back(static_cast<size_t>(stoull(x)));
    return out;
}

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        string key = argv[i];
        auto need = [&](const string &name) -> string {
            if (i + 1 >= argc) throw invalid_argument("Missing value for " + name);
            return string(argv[++i]);
        };
        if (key == "--poly_modulus_degree") args.poly_modulus_degree = static_cast<size_t>(stoull(need(key)));
        else if (key == "--coeff_mod_bits") args.coeff_mod_bits = parse_int_csv(need(key));
        else if (key == "--scale_bits") args.scale_bits = stod(need(key));
        else if (key == "--dims") args.dims = parse_size_t_csv(need(key));
        else if (key == "--repeats") args.repeats = static_cast<size_t>(stoull(need(key)));
        else if (key == "--warmups") args.warmups = static_cast<size_t>(stoull(need(key)));
        else if (key == "--typiclust_centroids") args.typiclust_centroids = static_cast<size_t>(stoull(need(key)));
        else if (key == "--output_csv") args.output_csv = need(key);
        else if (key == "--quiet") args.verbose = false;
        else throw invalid_argument("Unknown argument: " + key);
    }
    return args;
}

static vector<double> random_vector(size_t n, mt19937 &rng, double lo = -1.0, double hi = 1.0) {
    uniform_real_distribution<double> dist(lo, hi);
    vector<double> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

static double mean_of(const vector<double> &v) {
    if (v.empty()) return 0.0;
    return accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static TimerSummary summarize(const vector<double> &v) {
    TimerSummary s;
    if (v.empty()) return s;
    s.mean_ms = mean_of(v);
    double acc = 0.0;
    for (double x : v) acc += (x - s.mean_ms) * (x - s.mean_ms);
    s.std_ms = sqrt(acc / static_cast<double>(v.size()));
    s.min_ms = *min_element(v.begin(), v.end());
    s.max_ms = *max_element(v.begin(), v.end());
    return s;
}

static void rotate_sum_inplace(Ciphertext &ct, size_t used_slots, Evaluator &evaluator, const GaloisKeys &galois_keys) {
    for (size_t step = 1; step < used_slots; step <<= 1) {
        Ciphertext rotated;
        evaluator.rotate_vector(ct, static_cast<int>(step), galois_keys, rotated);
        evaluator.add_inplace(ct, rotated);
    }
}

static Plaintext encode_plain(const vector<double> &vals, double scale, CKKSEncoder &encoder) {
    Plaintext pt;
    encoder.encode(vals, scale, pt);
    return pt;
}

static Plaintext encode_plain_scalar(double val, double scale, CKKSEncoder &encoder) {
    Plaintext pt;
    encoder.encode(val, scale, pt);
    return pt;
}

static void modswitch_plain_to(const parms_id_type &pid, Plaintext &pt, Evaluator &evaluator) {
    evaluator.mod_switch_to_inplace(pt, pid);
}

static void align_cipher_plain(Ciphertext &ct, Plaintext &pt, Evaluator &evaluator, const SEALContext &context) {
    auto cidx = context.get_context_data(ct.parms_id())->chain_index();
    auto pidx = context.get_context_data(pt.parms_id())->chain_index();
    if (pidx > cidx) evaluator.mod_switch_to_inplace(pt, ct.parms_id());
    else if (cidx > pidx) evaluator.mod_switch_to_inplace(ct, pt.parms_id());
    pt.scale() = ct.scale();
}

static void align_cipher_cipher(Ciphertext &a, Ciphertext &b, Evaluator &evaluator, const SEALContext &context) {
    auto aidx = context.get_context_data(a.parms_id())->chain_index();
    auto bidx = context.get_context_data(b.parms_id())->chain_index();
    if (aidx > bidx) evaluator.mod_switch_to_inplace(a, b.parms_id());
    else if (bidx > aidx) evaluator.mod_switch_to_inplace(b, a.parms_id());
    b.scale() = a.scale();
}

static Ciphertext encrypt_vector(const vector<double> &vals, double scale, CKKSEncoder &encoder, Encryptor &encryptor) {
    Plaintext pt;
    encoder.encode(vals, scale, pt);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

static Ciphertext bench_cosine_core(
    const Ciphertext &enc_x,
    const Plaintext &plain_q,
    size_t dim,
    Evaluator &evaluator,
    const GaloisKeys &galois_keys
) {
    Ciphertext prod;
    evaluator.multiply_plain(enc_x, plain_q, prod);
    evaluator.rescale_to_next_inplace(prod);
    rotate_sum_inplace(prod, dim, evaluator, galois_keys);
    return prod;
}

static Ciphertext cubic_sigmoid_approx(
    Ciphertext x,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const SEALContext &context
) {
    // sigma(x) ≈ 0.5 + 0.197x - 0.004x^3
    double scale = x.scale();
    Plaintext p_half = encode_plain_scalar(0.5, scale, encoder);
    Plaintext p_a = encode_plain_scalar(0.197, scale, encoder);
    Plaintext p_b = encode_plain_scalar(-0.004, scale, encoder);

    Ciphertext x2, x3;
    evaluator.square(x, x2);
    evaluator.relinearize_inplace(x2, relin_keys);
    evaluator.rescale_to_next_inplace(x2);

    Ciphertext x_mod = x;
    evaluator.mod_switch_to_inplace(x_mod, x2.parms_id());
    x_mod.scale() = x2.scale();

    evaluator.multiply(x2, x_mod, x3);
    evaluator.relinearize_inplace(x3, relin_keys);
    evaluator.rescale_to_next_inplace(x3);

    Ciphertext ax, bx3;
    align_cipher_plain(x, p_a, evaluator, context);
    evaluator.multiply_plain(x, p_a, ax);
    evaluator.rescale_to_next_inplace(ax);

    align_cipher_plain(x3, p_b, evaluator, context);
    evaluator.multiply_plain(x3, p_b, bx3);
    evaluator.rescale_to_next_inplace(bx3);

    align_cipher_cipher(ax, bx3, evaluator, context);
    evaluator.add_inplace(ax, bx3);

    align_cipher_plain(ax, p_half, evaluator, context);
    evaluator.add_plain_inplace(ax, p_half);
    return ax;
}

static Ciphertext bench_entropy_core(
    const Ciphertext &enc_x,
    const Plaintext &plain_w,
    size_t dim,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const GaloisKeys &galois_keys,
    const RelinKeys &relin_keys,
    const SEALContext &context
) {
    Ciphertext logit = bench_cosine_core(enc_x, plain_w, dim, evaluator, galois_keys);
    Ciphertext p = cubic_sigmoid_approx(logit, encoder, evaluator, relin_keys, context);

    // entropy proxy: 4 p (1-p)
    Plaintext one = encode_plain_scalar(1.0, p.scale(), encoder);
    align_cipher_plain(p, one, evaluator, context);
    Ciphertext one_minus_p = p;
    evaluator.negate_inplace(one_minus_p);
    evaluator.add_plain_inplace(one_minus_p, one);

    align_cipher_cipher(p, one_minus_p, evaluator, context);
    Ciphertext out;
    evaluator.multiply(p, one_minus_p, out);
    evaluator.relinearize_inplace(out, relin_keys);
    evaluator.rescale_to_next_inplace(out);

    Plaintext four = encode_plain_scalar(4.0, out.scale(), encoder);
    align_cipher_plain(out, four, evaluator, context);
    evaluator.multiply_plain_inplace(out, four);
    evaluator.rescale_to_next_inplace(out);
    return out;
}

static Ciphertext bench_margin_core(
    const Ciphertext &enc_x,
    const Plaintext &plain_w,
    size_t dim,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const GaloisKeys &galois_keys,
    const RelinKeys &relin_keys,
    const SEALContext &context
) {
    Ciphertext logit = bench_cosine_core(enc_x, plain_w, dim, evaluator, galois_keys);
    Ciphertext p = cubic_sigmoid_approx(logit, encoder, evaluator, relin_keys, context);

    // margin proxy: 1 - 4(p-0.5)^2
    Plaintext half = encode_plain_scalar(0.5, p.scale(), encoder);
    align_cipher_plain(p, half, evaluator, context);
    evaluator.sub_plain_inplace(p, half);

    Ciphertext sq;
    evaluator.square(p, sq);
    evaluator.relinearize_inplace(sq, relin_keys);
    evaluator.rescale_to_next_inplace(sq);

    Plaintext four = encode_plain_scalar(4.0, sq.scale(), encoder);
    align_cipher_plain(sq, four, evaluator, context);
    evaluator.multiply_plain_inplace(sq, four);
    evaluator.rescale_to_next_inplace(sq);

    Plaintext one = encode_plain_scalar(1.0, sq.scale(), encoder);
    align_cipher_plain(sq, one, evaluator, context);
    evaluator.negate_inplace(sq);
    evaluator.add_plain_inplace(sq, one);
    return sq;
}

static Ciphertext bench_typiclust_core(
    const Ciphertext &enc_x,
    const vector<Plaintext> &plain_centroids,
    size_t dim,
    Evaluator &evaluator,
    const GaloisKeys &galois_keys,
    const SEALContext &context
) {
    // This benchmarks encrypted centroid-scoring only, not encrypted k-means fitting.
    Ciphertext accum;
    bool first = true;
    for (const auto &pt_raw : plain_centroids) {
        Plaintext pt = pt_raw;
        evaluator.mod_switch_to_inplace(pt, enc_x.parms_id());
        Ciphertext score = bench_cosine_core(enc_x, pt, dim, evaluator, galois_keys);
        if (first) {
            accum = score;
            first = false;
        } else {
            align_cipher_cipher(accum, score, evaluator, context);
            evaluator.add_inplace(accum, score);
        }
    }
    return accum;
}

static Ciphertext bench_ours_packaged_core(
    const Ciphertext &enc_img,
    const Ciphertext &enc_txt,
    const Plaintext &plain_qimg,
    const Plaintext &plain_w_match,
    const Plaintext &plain_w_qimg,
    const Plaintext &plain_w_quad,
    size_t dim,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const GaloisKeys &galois_keys,
    const RelinKeys &relin_keys,
    const SEALContext &context
) {
    // packaged quadratic scorer: dot(img,txt) + dot(img,qimg) + alpha * dot(img,qimg)^2
    Ciphertext match_ct;
    evaluator.multiply(enc_img, enc_txt, match_ct);
    evaluator.relinearize_inplace(match_ct, relin_keys);
    evaluator.rescale_to_next_inplace(match_ct);
    rotate_sum_inplace(match_ct, dim, evaluator, galois_keys);

    Ciphertext qimg_ct;
    evaluator.multiply_plain(enc_img, plain_qimg, qimg_ct);
    evaluator.rescale_to_next_inplace(qimg_ct);
    rotate_sum_inplace(qimg_ct, dim, evaluator, galois_keys);

    Ciphertext qimg_sq;
    evaluator.square(qimg_ct, qimg_sq);
    evaluator.relinearize_inplace(qimg_sq, relin_keys);
    evaluator.rescale_to_next_inplace(qimg_sq);

    Ciphertext match_w, qimg_w, quad_w;
    Plaintext pw1 = plain_w_match, pw2 = plain_w_qimg, pw3 = plain_w_quad;
    align_cipher_plain(match_ct, pw1, evaluator, context);
    evaluator.multiply_plain(match_ct, pw1, match_w);
    evaluator.rescale_to_next_inplace(match_w);

    align_cipher_plain(qimg_ct, pw2, evaluator, context);
    evaluator.multiply_plain(qimg_ct, pw2, qimg_w);
    evaluator.rescale_to_next_inplace(qimg_w);

    align_cipher_plain(qimg_sq, pw3, evaluator, context);
    evaluator.multiply_plain(qimg_sq, pw3, quad_w);
    evaluator.rescale_to_next_inplace(quad_w);

    align_cipher_cipher(match_w, qimg_w, evaluator, context);
    evaluator.add_inplace(match_w, qimg_w);
    align_cipher_cipher(match_w, quad_w, evaluator, context);
    evaluator.add_inplace(match_w, quad_w);
    return match_w;
}

static vector<double> measure_ms(const function<void()> &fn, size_t warmups, size_t repeats) {
    for (size_t i = 0; i < warmups; ++i) fn();
    vector<double> times;
    times.reserve(repeats);
    for (size_t i = 0; i < repeats; ++i) {
        auto t0 = chrono::high_resolution_clock::now();
        fn();
        auto t1 = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(t1 - t0).count());
    }
    return times;
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Args args = parse_args(argc, argv);
    double scale = pow(2.0, args.scale_bits);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(args.poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(args.poly_modulus_degree, args.coeff_mod_bits));
    SEALContext context(parms);

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    SecretKey secret_key = keygen.secret_key();
    RelinKeys relin_keys;
    GaloisKeys galois_keys;
    keygen.create_relin_keys(relin_keys);
    keygen.create_galois_keys(galois_keys);

    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    ofstream fout(args.output_csv);
    if (!fout) throw runtime_error("Cannot open output csv: " + args.output_csv);
    fout << "scheme,dim,repeats,warmups,mean_ms,std_ms,min_ms,max_ms,note\n";

    mt19937 rng(42);
    if (args.verbose) {
        cout << "poly_modulus_degree=" << args.poly_modulus_degree
             << " scale_bits=" << args.scale_bits
             << " repeats=" << args.repeats
             << " warmups=" << args.warmups << "\n";
    }

    for (size_t dim : args.dims) {
        vector<double> x = random_vector(dim, rng);
        vector<double> txt = random_vector(dim, rng);
        vector<double> q = random_vector(dim, rng);
        vector<double> qimg = random_vector(dim, rng);

        Ciphertext enc_x = encrypt_vector(x, scale, encoder, encryptor);
        Ciphertext enc_img = encrypt_vector(x, scale, encoder, encryptor);
        Ciphertext enc_txt = encrypt_vector(txt, scale, encoder, encryptor);
        Plaintext plain_q = encode_plain(q, scale, encoder);
        Plaintext plain_qimg = encode_plain(qimg, scale, encoder);
        evaluator.mod_switch_to_inplace(plain_q, enc_x.parms_id());
        evaluator.mod_switch_to_inplace(plain_qimg, enc_img.parms_id());

        Plaintext pw_match = encode_plain_scalar(0.42, scale, encoder);
        Plaintext pw_qimg = encode_plain_scalar(0.22, scale, encoder);
        Plaintext pw_quad = encode_plain_scalar(0.08, scale, encoder);
        evaluator.mod_switch_to_inplace(pw_match, enc_img.parms_id());
        evaluator.mod_switch_to_inplace(pw_qimg, enc_img.parms_id());
        evaluator.mod_switch_to_inplace(pw_quad, enc_img.parms_id());

        vector<Plaintext> centroid_pts;
        centroid_pts.reserve(args.typiclust_centroids);
        for (size_t c = 0; c < args.typiclust_centroids; ++c) {
            centroid_pts.push_back(encode_plain(random_vector(dim, rng), scale, encoder));
            evaluator.mod_switch_to_inplace(centroid_pts.back(), enc_x.parms_id());
        }

        vector<tuple<string, vector<double>, string>> rows;

        rows.push_back({"cosine_match_select",
            measure_ms([&]() {
                volatile Ciphertext out = bench_cosine_core(enc_x, plain_q, dim, evaluator, galois_keys);
                (void)out;
            }, args.warmups, args.repeats),
            "encrypted dot over candidate/query; eval only"});

        rows.push_back({"entropy_match_select",
            measure_ms([&]() {
                volatile Ciphertext out = bench_entropy_core(enc_x, plain_q, dim, encoder, evaluator, galois_keys, relin_keys, context);
                (void)out;
            }, args.warmups, args.repeats),
            "encrypted linear score + polynomial uncertainty proxy"});

        rows.push_back({"margin_match_select",
            measure_ms([&]() {
                volatile Ciphertext out = bench_margin_core(enc_x, plain_q, dim, encoder, evaluator, galois_keys, relin_keys, context);
                (void)out;
            }, args.warmups, args.repeats),
            "encrypted linear score + polynomial margin proxy"});

        rows.push_back({"typiclust_match_select",
            measure_ms([&]() {
                volatile Ciphertext out = bench_typiclust_core(enc_x, centroid_pts, dim, evaluator, galois_keys, context);
                (void)out;
            }, args.warmups, args.repeats),
            "encrypted centroid scoring only; excludes clustering fit"});

        rows.push_back({"ours_main_select",
            measure_ms([&]() {
                volatile Ciphertext out = bench_ours_packaged_core(enc_img, enc_txt, plain_qimg, pw_match, pw_qimg, pw_quad,
                    dim, encoder, evaluator, galois_keys, relin_keys, context);
                (void)out;
            }, args.warmups, args.repeats),
            "buyer-conditioned packaged quadratic scorer; eval only"});

        for (auto &row : rows) {
            const string &scheme = get<0>(row);
            const vector<double> &vals = get<1>(row);
            const string &note = get<2>(row);
            TimerSummary s = summarize(vals);
            fout << scheme << ',' << dim << ',' << args.repeats << ',' << args.warmups << ','
                 << fixed << setprecision(6)
                 << s.mean_ms << ',' << s.std_ms << ',' << s.min_ms << ',' << s.max_ms << ',' << '"' << note << '"' << '\n';
            if (args.verbose) {
                cout << left << setw(26) << scheme
                     << " dim=" << setw(5) << dim
                     << " mean_ms=" << setw(10) << s.mean_ms
                     << " std_ms=" << setw(10) << s.std_ms
                     << " note=" << note << '\n';
            }
        }
    }

    if (args.verbose) cout << "Saved CSV to " << args.output_csv << "\n";
    return 0;
}
