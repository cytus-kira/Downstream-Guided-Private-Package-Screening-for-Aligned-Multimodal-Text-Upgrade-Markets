#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace std;
using namespace seal;

struct Args {
    size_t batch = 32;
    size_t dim = 512;
    size_t repeats = 5;
    size_t warmups = 1;
    uint64_t seed = 123456;
    string output_csv = "results/packing_ablation.csv";
    string output_tex = "results/packing_ablation_table4.tex";
};

struct Summary {
    double mean{0.0};
    double std{0.0};
};

static Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        string key = argv[i];
        auto need = [&](const string &k) -> string {
            if (i + 1 >= argc) throw runtime_error("Missing value for " + k);
            return argv[++i];
        };
        if (key == "--batch") args.batch = static_cast<size_t>(stoull(need(key)));
        else if (key == "--dim") args.dim = static_cast<size_t>(stoull(need(key)));
        else if (key == "--repeats") args.repeats = static_cast<size_t>(stoull(need(key)));
        else if (key == "--warmups") args.warmups = static_cast<size_t>(stoull(need(key)));
        else if (key == "--seed") args.seed = static_cast<uint64_t>(stoull(need(key)));
        else if (key == "--output_csv") args.output_csv = need(key);
        else if (key == "--output_tex") args.output_tex = need(key);
        else throw runtime_error("Unknown argument: " + key);
    }
    return args;
}

static bool is_power_of_two(size_t x) {
    return x && ((x & (x - 1)) == 0);
}

static size_t ilog2_exact(size_t x) {
    if (!is_power_of_two(x)) throw runtime_error("dimension must be a power of two");
    size_t r = 0;
    while ((size_t(1) << r) < x) ++r;
    return r;
}

static Summary summarize(const vector<double> &xs) {
    if (xs.empty()) return {};
    double mean = accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
    double var = 0.0;
    for (double x : xs) var += (x - mean) * (x - mean);
    return {mean, sqrt(var / static_cast<double>(xs.size()))};
}

static string fmt(double x) {
    ostringstream os;
    os << fixed << setprecision(6) << x;
    return os.str();
}

static void ensure_parent_dir(const string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == string::npos) return;
    string dir = path.substr(0, pos);
    if (dir.empty()) return;
    string cur;
    for (char c : dir) {
        cur.push_back(c);
        if (c == '/' || c == '\\') {
            if (cur.size() > 1) mkdir(cur.c_str(), 0777);
        }
    }
    mkdir(dir.c_str(), 0777);
}

static double max_abs_error(const vector<double> &a, const vector<double> &b) {
    double m = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = max(m, fabs(a[i] - b[i]));
    return m;
}

static double rel_l2(const vector<double> &a, const vector<double> &b) {
    double diff2 = 0.0;
    double ref2 = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        diff2 += d * d;
        ref2 += a[i] * a[i];
    }
    return sqrt(diff2) / (sqrt(ref2) + 1e-12);
}

// From the previous main_fast.cpp path: A is row-major A[i*N+j].
static void random_matrix(vector<double> &A, size_t N, mt19937_64 &rng) {
    uniform_real_distribution<double> dist(-1.0, 1.0);
    A.resize(N * N);
    for (double &v : A) v = dist(rng);
}

// From main_fast.cpp: x is row-major x[b*N+j].
static void random_batched_vector(vector<double> &x, size_t B, size_t N, mt19937_64 &rng) {
    uniform_real_distribution<double> dist(-1.0, 1.0);
    x.resize(B * N);
    for (double &v : x) v = dist(rng);
}

static void transpose_matrix(const vector<double> &A, size_t N, vector<double> &A_T) {
    A_T.assign(N * N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_T[j * N + i] = A[i * N + j];
}

// y[b,i] = sum_j A[i,j] * x[b,j].
static void plain_matvec(const vector<double> &A, const vector<double> &x,
                         size_t B, size_t N, vector<double> &y) {
    y.assign(B * N, 0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < N; ++j) s += A[i * N + j] * x[b * N + j];
            y[b * N + i] = s;
        }
    }
}

// From main_fast.cpp: pack x[b,j] to slot t=j*B+b.
static void pack_x(const vector<double> &x, size_t B, size_t N, vector<double> &x_packed) {
    x_packed.assign(B * N, 0.0);
    for (size_t b = 0; b < B; ++b)
        for (size_t j = 0; j < N; ++j)
            x_packed[j * B + b] = x[b * N + j];
}

// From main_fast.cpp diagonal encoding:
// slot t corresponds to output coordinate i and sample b: t=i*B+b.
// rotate by k*B makes rotated[t] = x[b,(i+k) mod N].
// diag_data[k][t] = A[i,(i+k) mod N].
static void build_diagonal_data(const vector<double> &A, size_t B, size_t N,
                                vector<vector<double>> &diag_data) {
    size_t slot_count = B * N;
    diag_data.assign(N, vector<double>(slot_count, 0.0));
    for (size_t k = 0; k < N; ++k) {
        vector<double> &diag_k = diag_data[k];
        for (size_t i = 0; i < N; ++i) {
            size_t j = (i + k) % N;
            double a_ij = A[i * N + j];
            for (size_t b = 0; b < B; ++b) diag_k[i * B + b] = a_ij;
        }
    }
}

static void plain_diag_batched(const vector<vector<double>> &diag_data,
                               const vector<double> &x_packed,
                               size_t B, size_t N,
                               vector<double> &y_packed) {
    size_t slot_count = B * N;
    y_packed.assign(slot_count, 0.0);
    vector<double> rotated(slot_count);
    for (size_t k = 0; k < N; ++k) {
        size_t step = (k * B) % slot_count;
        for (size_t t = 0; t < slot_count; ++t) rotated[t] = x_packed[(t + step) % slot_count];
        const vector<double> &diag_k = diag_data[k];
        for (size_t t = 0; t < slot_count; ++t) y_packed[t] += diag_k[t] * rotated[t];
    }
}

static void unpack_y(const vector<double> &y_packed, size_t B, size_t N, vector<double> &y) {
    y.assign(B * N, 0.0);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < N; ++i)
            y[b * N + i] = y_packed[i * B + b];
}

static void ckks_diag_matvec_batched(const SEALContext &context,
                                     CKKSEncoder &encoder,
                                     Evaluator &evaluator,
                                     const GaloisKeys &gal_keys,
                                     const Ciphertext &enc_x,
                                     const vector<Plaintext> &diag_plains,
                                     size_t B, size_t N,
                                     Ciphertext &enc_y) {
    size_t slot_count = encoder.slot_count();
    if (slot_count < B * N) throw runtime_error("slot_count < B*N");

    bool first_term = true;
    Ciphertext rotated;
    Ciphertext tmp;
    for (size_t k = 0; k < N; ++k) {
        int step = static_cast<int>((k * B) % slot_count);
        if (k == 0) rotated = enc_x;
        else evaluator.rotate_vector(enc_x, step, gal_keys, rotated);

        evaluator.multiply_plain(rotated, diag_plains[k], tmp);
        evaluator.rescale_to_next_inplace(tmp);

        if (first_term) {
            enc_y = tmp;
            first_term = false;
        } else {
            if (enc_y.parms_id() != tmp.parms_id()) {
                evaluator.mod_switch_to_inplace(enc_y, tmp.parms_id());
            }
            if (fabs(log2(enc_y.scale()) - log2(tmp.scale())) > 1.0) enc_y.scale() = tmp.scale();
            evaluator.add_inplace(enc_y, tmp);
        }
    }
}

static vector<Plaintext> encode_diagonal_plains(const vector<vector<double>> &diag_data,
                                                double scale,
                                                CKKSEncoder &encoder) {
    vector<Plaintext> plains(diag_data.size());
    for (size_t k = 0; k < diag_data.size(); ++k) encoder.encode(diag_data[k], scale, plains[k]);
    return plains;
}

static vector<double> decrypt_active(const Ciphertext &ct, size_t active,
                                     Decryptor &decryptor, CKKSEncoder &encoder) {
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    vector<complex<double>> decoded;
    encoder.decode(pt, decoded);
    vector<double> out(active);
    for (size_t i = 0; i < active; ++i) out[i] = decoded[i].real();
    return out;
}

static size_t naive_rotation_count(size_t dim) {
    return dim * ilog2_exact(dim) + (dim - 1);
}

static size_t proposed_rotation_count(size_t dim) {
    return dim - 1;
}

static void write_outputs(const Args &args,
                          size_t naive_rot,
                          size_t proposed_rot,
                          const Summary &fwd,
                          const Summary &bwd,
                          const Summary &total,
                          double fwd_rel_plain,
                          double bwd_rel_plain,
                          double fwd_rel_ckks,
                          double bwd_rel_ckks) {
    ensure_parent_dir(args.output_csv);
    ofstream csv(args.output_csv);
    if (!csv) throw runtime_error("Cannot open CSV output");
    csv << "packing,forward_rotations,backward_rotations,total_rotations,"
        << "forward_time_mean_s,forward_time_std_s,"
        << "backward_time_mean_s,backward_time_std_s,"
        << "total_time_mean_s,total_time_std_s,"
        << "forward_rel_l2_plain,backward_rel_l2_plain,"
        << "forward_rel_l2_ckks,backward_rel_l2_ckks,note\n";
    csv << "NaivePack," << naive_rot << ',' << naive_rot << ',' << 2 * naive_rot
        << ",nan,nan,nan,nan,nan,nan,0,0,nan,nan,"
        << "\"operation-count only; direct CKKS Naive runtime not executed\"\n";
    csv << "ProposedPack," << proposed_rot << ',' << proposed_rot << ',' << 2 * proposed_rot
        << ',' << fmt(fwd.mean) << ',' << fmt(fwd.std)
        << ',' << fmt(bwd.mean) << ',' << fmt(bwd.std)
        << ',' << fmt(total.mean) << ',' << fmt(total.std)
        << ',' << fwd_rel_plain << ',' << bwd_rel_plain
        << ',' << fwd_rel_ckks << ',' << bwd_rel_ckks
        << ",\"main_fast row/diagonal CKKS runtime\"\n";

    ensure_parent_dir(args.output_tex);
    ofstream tex(args.output_tex);
    if (!tex) throw runtime_error("Cannot open TeX output");
    tex << "\\begin{table}[H]\n"
        << "\\caption{Rotation-count ablation of the \\csfuse packed execution under the same guarded-hybrid boundary and CKKS configuration.}\n"
        << "\\label{tab:packing-rotation-ablation}\n"
        << "\\centering\n"
        << "\\pftablestyle\n"
        << "\\setlength{\\tabcolsep}{3pt}\n"
        << "\\begin{tabular}{@{}lccc@{}}\n"
        << "\\toprule\n"
        << "Packing & Fwd. rotations & Bwd. rotations & Total rotations \\\\\n"
        << "\\midrule\n"
        << "NaivePack    & " << naive_rot << " & " << naive_rot << " & " << 2 * naive_rot << " \\\\\n"
        << "ProposedPack & " << proposed_rot << " & " << proposed_rot << " & " << 2 * proposed_rot << " \\\\\n"
        << "\\bottomrule\n"
        << "\\end{tabular}\n"
        << "\\end{table}\n";
}

int main(int argc, char **argv) {
    cout << unitbuf;
    Args args = parse_args(argc, argv);
    if (!is_power_of_two(args.dim)) throw runtime_error("--dim must be a power of two");
    if (args.batch * args.dim != 16384) {
        throw runtime_error("This main_fast-derived harness expects batch*dim=16384");
    }

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(32768);
    parms.set_coeff_modulus(CoeffModulus::Create(32768, {60, 40, 40, 40}));
    SEALContext context(parms);
    if (!context.parameters_set()) {
        throw runtime_error(string("Invalid CKKS parameters: ") + context.parameter_error_message());
    }

    CKKSEncoder encoder(context);
    if (encoder.slot_count() != args.batch * args.dim) {
        throw runtime_error("Unexpected slot_count for requested batch/dim");
    }

    KeyGenerator keygen(context);
    PublicKey public_key;
    keygen.create_public_key(public_key);
    SecretKey secret_key = keygen.secret_key();
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);

    mt19937_64 rng(args.seed);
    vector<double> A;
    vector<double> x;
    vector<double> dy;
    random_matrix(A, args.dim, rng);
    random_batched_vector(x, args.batch, args.dim, rng);
    random_batched_vector(dy, args.batch, args.dim, rng);

    vector<double> A_T;
    transpose_matrix(A, args.dim, A_T);

    vector<double> y_plain;
    vector<double> dx_plain;
    plain_matvec(A, x, args.batch, args.dim, y_plain);
    plain_matvec(A_T, dy, args.batch, args.dim, dx_plain);

    vector<double> x_packed;
    vector<double> dy_packed;
    pack_x(x, args.batch, args.dim, x_packed);
    pack_x(dy, args.batch, args.dim, dy_packed);

    vector<vector<double>> diag_A;
    vector<vector<double>> diag_AT;
    build_diagonal_data(A, args.batch, args.dim, diag_A);
    build_diagonal_data(A_T, args.batch, args.dim, diag_AT);

    vector<double> y_diag_packed;
    vector<double> dx_diag_packed;
    vector<double> y_diag;
    vector<double> dx_diag;
    plain_diag_batched(diag_A, x_packed, args.batch, args.dim, y_diag_packed);
    plain_diag_batched(diag_AT, dy_packed, args.batch, args.dim, dx_diag_packed);
    unpack_y(y_diag_packed, args.batch, args.dim, y_diag);
    unpack_y(dx_diag_packed, args.batch, args.dim, dx_diag);
    double fwd_rel_plain = rel_l2(y_plain, y_diag);
    double bwd_rel_plain = rel_l2(dx_plain, dx_diag);

    double scale = pow(2.0, 40);
    Plaintext pt_x;
    Plaintext pt_dy;
    encoder.encode(x_packed, scale, pt_x);
    encoder.encode(dy_packed, scale, pt_dy);
    Ciphertext enc_x;
    Ciphertext enc_dy;
    encryptor.encrypt(pt_x, enc_x);
    encryptor.encrypt(pt_dy, enc_dy);

    auto diag_A_pt = encode_diagonal_plains(diag_A, scale, encoder);
    auto diag_AT_pt = encode_diagonal_plains(diag_AT, scale, encoder);

    auto run_forward = [&]() {
        Ciphertext out;
        ckks_diag_matvec_batched(context, encoder, evaluator, gal_keys,
                                 enc_x, diag_A_pt, args.batch, args.dim, out);
        return out;
    };
    auto run_backward = [&]() {
        Ciphertext out;
        ckks_diag_matvec_batched(context, encoder, evaluator, gal_keys,
                                 enc_dy, diag_AT_pt, args.batch, args.dim, out);
        return out;
    };

    Ciphertext fwd_check = run_forward();
    Ciphertext bwd_check = run_backward();
    auto y_he_packed = decrypt_active(fwd_check, args.batch * args.dim, decryptor, encoder);
    auto dx_he_packed = decrypt_active(bwd_check, args.batch * args.dim, decryptor, encoder);
    vector<double> y_he;
    vector<double> dx_he;
    unpack_y(y_he_packed, args.batch, args.dim, y_he);
    unpack_y(dx_he_packed, args.batch, args.dim, dx_he);
    double fwd_rel_ckks = rel_l2(y_plain, y_he);
    double bwd_rel_ckks = rel_l2(dx_plain, dx_he);

    for (size_t i = 0; i < args.warmups; ++i) {
        volatile size_t keep = run_forward().size() + run_backward().size();
        (void)keep;
    }

    vector<double> fwd_times;
    vector<double> bwd_times;
    vector<double> total_times;
    for (size_t i = 0; i < args.repeats; ++i) {
        auto t0 = chrono::high_resolution_clock::now();
        auto f = run_forward();
        auto t1 = chrono::high_resolution_clock::now();
        auto b = run_backward();
        auto t2 = chrono::high_resolution_clock::now();
        volatile size_t keep = f.size() + b.size();
        (void)keep;
        double fs = chrono::duration<double>(t1 - t0).count();
        double bs = chrono::duration<double>(t2 - t1).count();
        fwd_times.push_back(fs);
        bwd_times.push_back(bs);
        total_times.push_back(fs + bs);
    }

    size_t naive_rot = naive_rotation_count(args.dim);
    size_t proposed_rot = proposed_rotation_count(args.dim);
    Summary fwd = summarize(fwd_times);
    Summary bwd = summarize(bwd_times);
    Summary total = summarize(total_times);
    write_outputs(args, naive_rot, proposed_rot, fwd, bwd, total,
                  fwd_rel_plain, bwd_rel_plain, fwd_rel_ckks, bwd_rel_ckks);

    cout << "[config] source=main_fast row/diagonal batch=" << args.batch
         << " dim=" << args.dim
         << " repeats=" << args.repeats
         << " warmups=" << args.warmups
         << " slot_count=" << encoder.slot_count()
         << "\n";
    cout << "[verify-plain] forward_rel_l2=" << fwd_rel_plain
         << " backward_rel_l2=" << bwd_rel_plain
         << " max_forward_abs=" << max_abs_error(y_plain, y_diag)
         << "\n";
    cout << "[verify-ckks] forward_rel_l2=" << fwd_rel_ckks
         << " backward_rel_l2=" << bwd_rel_ckks
         << "\n";
    cout << "[rotations] NaivePack=" << 2 * naive_rot
         << " ProposedPack=" << 2 * proposed_rot
         << " reduction=" << fixed << setprecision(2)
         << 100.0 * (1.0 - double(proposed_rot) / double(naive_rot)) << "%\n";
    cout << "[runtime] ProposedPack forward_mean_s=" << fmt(fwd.mean)
         << " backward_mean_s=" << fmt(bwd.mean)
         << " total_mean_s=" << fmt(total.mean)
         << "\n";
    cout << "[saved] " << args.output_csv << "\n";
    cout << "[saved] " << args.output_tex << "\n";
    return 0;
}
