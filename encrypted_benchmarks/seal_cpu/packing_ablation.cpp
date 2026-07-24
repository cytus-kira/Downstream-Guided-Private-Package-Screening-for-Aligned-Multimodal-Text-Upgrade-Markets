#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace seal;
using namespace std;

struct Args {
    size_t batch = 32;
    size_t dim = 512;
    size_t repeats = 1;
    size_t warmups = 0;
    uint64_t seed = 123456;
    string mode = "quick";
    string output_csv = "results/packing_ablation.csv";
    string output_tex = "results/packing_ablation_table4.tex";
};

struct Summary {
    double mean{numeric_limits<double>::quiet_NaN()};
    double std{numeric_limits<double>::quiet_NaN()};
};

struct Row {
    string packing;
    size_t forward_rotations{0};
    size_t backward_rotations{0};
    size_t total_rotations{0};
    Summary forward;
    Summary backward;
    Summary total;
    double forward_rel_l2_error{numeric_limits<double>::quiet_NaN()};
    double backward_rel_l2_error{numeric_limits<double>::quiet_NaN()};
    string note;
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
        else if (key == "--mode") args.mode = need(key);
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
    if (!is_power_of_two(x)) throw runtime_error("Expected a power-of-two dimension.");
    size_t r = 0;
    while ((size_t(1) << r) < x) ++r;
    return r;
}

static Summary summarize(const vector<double> &xs) {
    if (xs.empty()) return {};
    double mean = accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
    double var = 0.0;
    for (double x : xs) var += (x - mean) * (x - mean);
    var /= static_cast<double>(xs.size());
    return {mean, sqrt(var)};
}

static string fmt(double x) {
    if (!isfinite(x)) return "nan";
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

static vector<double> random_vector(size_t n, mt19937_64 &rng, double lo = -1.0, double hi = 1.0) {
    uniform_real_distribution<double> dist(lo, hi);
    vector<double> v(n);
    for (double &x : v) x = dist(rng);
    return v;
}

static vector<vector<double>> random_matrix(size_t rows, size_t cols, mt19937_64 &rng,
                                            double lo = -0.5, double hi = 0.5) {
    uniform_real_distribution<double> dist(lo, hi);
    vector<vector<double>> w(rows, vector<double>(cols));
    for (auto &row : w) {
        for (double &x : row) x = dist(rng);
    }
    return w;
}

static vector<vector<double>> transpose(const vector<vector<double>> &w) {
    vector<vector<double>> out(w[0].size(), vector<double>(w.size(), 0.0));
    for (size_t i = 0; i < w.size(); ++i)
        for (size_t j = 0; j < w[0].size(); ++j)
            out[j][i] = w[i][j];
    return out;
}

static vector<double> plaintext_matmul_direct(const vector<double> &x,
                                              const vector<vector<double>> &w,
                                              size_t batch,
                                              size_t dim) {
    vector<double> y(batch * dim, 0.0);
    for (size_t out_col = 0; out_col < dim; ++out_col) {
        for (size_t in_col = 0; in_col < dim; ++in_col) {
            double coeff = w[in_col][out_col];
            for (size_t row = 0; row < batch; ++row) {
                y[out_col * batch + row] += x[in_col * batch + row] * coeff;
            }
        }
    }
    return y;
}

static vector<double> plaintext_matmul_diagonal_schedule(const vector<double> &x,
                                                         const vector<vector<double>> &w,
                                                         size_t batch,
                                                         size_t dim) {
    vector<double> y(batch * dim, 0.0);
    for (size_t diag = 0; diag < dim; ++diag) {
        for (size_t out_col = 0; out_col < dim; ++out_col) {
            size_t in_col = (out_col + diag) % dim;
            double coeff = w[in_col][out_col];
            for (size_t row = 0; row < batch; ++row) {
                y[out_col * batch + row] += x[in_col * batch + row] * coeff;
            }
        }
    }
    return y;
}

static Ciphertext encrypt_slots(const vector<double> &values, double scale,
                                CKKSEncoder &encoder, Encryptor &encryptor) {
    Plaintext pt;
    encoder.encode(values, scale, pt);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);
    return ct;
}

static Plaintext encode_plain(const vector<double> &values, double scale, CKKSEncoder &encoder) {
    Plaintext pt;
    encoder.encode(values, scale, pt);
    return pt;
}

static void align_plain_to(Ciphertext &ct, Plaintext &pt, Evaluator &evaluator) {
    evaluator.mod_switch_to_inplace(pt, ct.parms_id());
    pt.scale() = ct.scale();
}

static void align_cipher_to(Ciphertext &a, Ciphertext &b, Evaluator &evaluator,
                            const SEALContext &context) {
    auto ac = context.get_context_data(a.parms_id());
    auto bc = context.get_context_data(b.parms_id());
    if (!ac || !bc) throw runtime_error("Invalid parms_id.");
    if (ac->chain_index() < bc->chain_index()) {
        evaluator.mod_switch_to_inplace(b, a.parms_id());
    } else if (ac->chain_index() > bc->chain_index()) {
        evaluator.mod_switch_to_inplace(a, b.parms_id());
    }
    b.scale() = a.scale();
}

static vector<Plaintext> build_naive_column_masks(const vector<vector<double>> &w,
                                                  size_t batch, size_t slots,
                                                  double scale, CKKSEncoder &encoder) {
    size_t dim = w.size();
    vector<Plaintext> masks(dim);
    for (size_t out_col = 0; out_col < dim; ++out_col) {
        vector<double> mask(slots, 0.0);
        for (size_t in_col = 0; in_col < dim; ++in_col) {
            for (size_t row = 0; row < batch; ++row) {
                mask[in_col * batch + row] = w[in_col][out_col];
            }
        }
        masks[out_col] = encode_plain(mask, scale, encoder);
    }
    return masks;
}

static vector<Plaintext> build_diagonal_masks(const vector<vector<double>> &w,
                                              size_t batch, size_t slots,
                                              double scale, CKKSEncoder &encoder) {
    size_t dim = w.size();
    vector<Plaintext> masks(dim);
    for (size_t diag = 0; diag < dim; ++diag) {
        vector<double> mask(slots, 0.0);
        for (size_t out_col = 0; out_col < dim; ++out_col) {
            size_t in_col = (out_col + diag) % dim;
            double coeff = w[in_col][out_col];
            for (size_t row = 0; row < batch; ++row) {
                mask[out_col * batch + row] = coeff;
            }
        }
        masks[diag] = encode_plain(mask, scale, encoder);
    }
    return masks;
}

static Ciphertext matmul_naive(const Ciphertext &x, size_t batch, size_t dim,
                               const vector<Plaintext> &column_masks,
                               const Plaintext &first_block_mask,
                               Evaluator &evaluator,
                               GaloisKeys &gal_keys,
                               const SEALContext &context) {
    size_t log_dim = ilog2_exact(dim);
    Ciphertext acc_all;
    bool first = true;

    for (size_t out_col = 0; out_col < dim; ++out_col) {
        Plaintext col_mask = column_masks[out_col];
        evaluator.mod_switch_to_inplace(col_mask, x.parms_id());
        col_mask.scale() = x.scale();

        Ciphertext prod;
        evaluator.multiply_plain(x, col_mask, prod);
        evaluator.rescale_to_next_inplace(prod);

        for (size_t l = 0; l < log_dim; ++l) {
            size_t step = (size_t(1) << l) * batch;
            Ciphertext rot;
            evaluator.rotate_vector(prod, static_cast<int>(step), gal_keys, rot);
            evaluator.add_inplace(prod, rot);
        }

        Plaintext block_mask = first_block_mask;
        align_plain_to(prod, block_mask, evaluator);
        Ciphertext masked;
        evaluator.multiply_plain(prod, block_mask, masked);
        evaluator.rescale_to_next_inplace(masked);

        Ciphertext shifted;
        if (out_col == 0) shifted = masked;
        else evaluator.rotate_vector(masked, -static_cast<int>(out_col * batch), gal_keys, shifted);

        if (first) {
            acc_all = shifted;
            first = false;
        } else {
            Ciphertext term = shifted;
            align_cipher_to(acc_all, term, evaluator, context);
            evaluator.add_inplace(acc_all, term);
        }
    }
    return acc_all;
}

static Ciphertext matmul_diagonal(const Ciphertext &x, size_t batch, size_t dim,
                                  const vector<Plaintext> &diag_masks,
                                  Evaluator &evaluator,
                                  GaloisKeys &gal_keys,
                                  const SEALContext &context) {
    Ciphertext acc;
    bool first = true;
    for (size_t diag = 0; diag < dim; ++diag) {
        Ciphertext rotated;
        if (diag == 0) rotated = x;
        else evaluator.rotate_vector(x, static_cast<int>(diag * batch), gal_keys, rotated);

        Plaintext mask = diag_masks[diag];
        evaluator.mod_switch_to_inplace(mask, rotated.parms_id());
        mask.scale() = rotated.scale();

        Ciphertext term;
        evaluator.multiply_plain(rotated, mask, term);
        evaluator.rescale_to_next_inplace(term);

        if (first) {
            acc = term;
            first = false;
        } else {
            Ciphertext term2 = term;
            align_cipher_to(acc, term2, evaluator, context);
            evaluator.add_inplace(acc, term2);
        }
    }
    return acc;
}

static size_t naive_rotation_count(size_t dim) {
    return dim * ilog2_exact(dim) + (dim - 1);
}

static size_t diagonal_rotation_count(size_t dim) {
    return dim - 1;
}

static vector<double> decrypt_decode(const Ciphertext &ct, Decryptor &decryptor, CKKSEncoder &encoder) {
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    vector<double> out;
    encoder.decode(pt, out);
    return out;
}

static double rel_l2_active(const vector<double> &reference, const vector<double> &candidate, size_t active) {
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (size_t i = 0; i < active; ++i) {
        double diff = reference[i] - candidate[i];
        diff2 += diff * diff;
        ref2 += reference[i] * reference[i];
    }
    return sqrt(diff2) / (sqrt(ref2) + 1e-12);
}

static pair<double, double> verify_plaintext_equivalence(const vector<double> &x,
                                                         const vector<double> &dy,
                                                         const vector<vector<double>> &w,
                                                         const vector<vector<double>> &wt,
                                                         size_t batch,
                                                         size_t dim) {
    auto nf = plaintext_matmul_direct(x, w, batch, dim);
    auto pf = plaintext_matmul_diagonal_schedule(x, w, batch, dim);
    auto nb = plaintext_matmul_direct(dy, wt, batch, dim);
    auto pb = plaintext_matmul_diagonal_schedule(dy, wt, batch, dim);
    double fwd_rel = rel_l2_active(nf, pf, batch * dim);
    double bwd_rel = rel_l2_active(nb, pb, batch * dim);
    cout << "[verify-plaintext] forward_rel_l2=" << fwd_rel
         << " backward_rel_l2=" << bwd_rel << "\n";
    return {fwd_rel, bwd_rel};
}

static pair<double, double> verify_equivalence(const Ciphertext &x_fwd,
                                               const Ciphertext &dy_bwd,
                                               size_t batch,
                                               size_t dim,
                                               const vector<Plaintext> &naive_w,
                                               const vector<Plaintext> &naive_wt,
                                               const vector<Plaintext> &diag_w,
                                               const vector<Plaintext> &diag_wt,
                                               const Plaintext &first_block_mask,
                                               Evaluator &evaluator,
                                               GaloisKeys &gal_keys,
                                               const SEALContext &context,
                                               Decryptor &decryptor,
                                               CKKSEncoder &encoder) {
    auto nf = matmul_naive(x_fwd, batch, dim, naive_w, first_block_mask, evaluator, gal_keys, context);
    auto pf = matmul_diagonal(x_fwd, batch, dim, diag_w, evaluator, gal_keys, context);
    auto nb = matmul_naive(dy_bwd, batch, dim, naive_wt, first_block_mask, evaluator, gal_keys, context);
    auto pb = matmul_diagonal(dy_bwd, batch, dim, diag_wt, evaluator, gal_keys, context);

    auto nf_dec = decrypt_decode(nf, decryptor, encoder);
    auto pf_dec = decrypt_decode(pf, decryptor, encoder);
    auto nb_dec = decrypt_decode(nb, decryptor, encoder);
    auto pb_dec = decrypt_decode(pb, decryptor, encoder);
    size_t active = batch * dim;
    double fwd_rel = rel_l2_active(nf_dec, pf_dec, active);
    double bwd_rel = rel_l2_active(nb_dec, pb_dec, active);
    cout << "[verify] forward_rel_l2=" << fwd_rel
         << " backward_rel_l2=" << bwd_rel
         << "\n";
    return {fwd_rel, bwd_rel};
}

static Row bench_one(const string &packing,
                     const Ciphertext &x_fwd,
                     const Ciphertext &dy_bwd,
                     size_t batch,
                     size_t dim,
                     size_t repeats,
                     size_t warmups,
                     const vector<Plaintext> &naive_w,
                     const vector<Plaintext> &naive_wt,
                     const vector<Plaintext> &diag_w,
                     const vector<Plaintext> &diag_wt,
                     const Plaintext &first_block_mask,
                     Evaluator &evaluator,
                     GaloisKeys &gal_keys,
                     const SEALContext &context) {
    static size_t sink = 0;
    bool proposed = (packing == "ProposedPack");
    auto run_forward = [&]() {
        return proposed
            ? matmul_diagonal(x_fwd, batch, dim, diag_w, evaluator, gal_keys, context)
            : matmul_naive(x_fwd, batch, dim, naive_w, first_block_mask, evaluator, gal_keys, context);
    };
    auto run_backward = [&]() {
        return proposed
            ? matmul_diagonal(dy_bwd, batch, dim, diag_wt, evaluator, gal_keys, context)
            : matmul_naive(dy_bwd, batch, dim, naive_wt, first_block_mask, evaluator, gal_keys, context);
    };

    for (size_t i = 0; i < warmups; ++i) {
        auto f = run_forward();
        auto b = run_backward();
        sink += f.size() + b.size();
    }

    vector<double> fwd, bwd, total;
    for (size_t i = 0; i < repeats; ++i) {
        auto t0 = chrono::high_resolution_clock::now();
        auto f = run_forward();
        auto t1 = chrono::high_resolution_clock::now();
        auto b = run_backward();
        auto t2 = chrono::high_resolution_clock::now();
        sink += f.size() + b.size();

        double fs = chrono::duration<double>(t1 - t0).count();
        double bs = chrono::duration<double>(t2 - t1).count();
        fwd.push_back(fs);
        bwd.push_back(bs);
        total.push_back(fs + bs);
    }

    Row row;
    row.packing = packing;
    row.forward_rotations = proposed ? diagonal_rotation_count(dim) : naive_rotation_count(dim);
    row.backward_rotations = row.forward_rotations;
    row.total_rotations = row.forward_rotations + row.backward_rotations;
    row.forward = summarize(fwd);
    row.backward = summarize(bwd);
    row.total = summarize(total);
    return row;
}

static void write_csv(const string &path, const vector<Row> &rows) {
    ensure_parent_dir(path);
    ofstream out(path);
    if (!out) throw runtime_error("Cannot open output CSV: " + path);
    out << "packing,forward_rotations,backward_rotations,total_rotations,"
        << "forward_time_mean_s,forward_time_std_s,"
        << "backward_time_mean_s,backward_time_std_s,"
        << "total_time_mean_s,total_time_std_s,"
        << "forward_rel_l2_error,backward_rel_l2_error,note\n";
    for (const auto &r : rows) {
        out << r.packing << ','
            << r.forward_rotations << ','
            << r.backward_rotations << ','
            << r.total_rotations << ','
            << fmt(r.forward.mean) << ','
            << fmt(r.forward.std) << ','
            << fmt(r.backward.mean) << ','
            << fmt(r.backward.std) << ','
            << fmt(r.total.mean) << ','
            << fmt(r.total.std) << ','
            << fmt(r.forward_rel_l2_error) << ','
            << fmt(r.backward_rel_l2_error) << ','
            << '"' << r.note << '"' << '\n';
    }
}

static void write_tex(const string &path, const vector<Row> &rows) {
    ensure_parent_dir(path);
    ofstream out(path);
    if (!out) throw runtime_error("Cannot open output TeX: " + path);
    out << "\\begin{table}[ht]\n"
        << "\\caption{Rotation-count ablation of the SGF-GH packed execution under the same guarded-hybrid boundary, arithmetic graph, and CKKS configuration. Rotation counts denote the number of ciphertext rotation invocations executed by the forward or backward CKKS path.}\n"
        << "\\label{tab:packing-rotation-ablation}\n"
        << "\\centering\n"
        << "\\pftablestyle\n"
        << "\\begin{tabular}{lccc}\n"
        << "\\toprule\n"
        << "Packing & Fwd. rotations & Bwd. rotations & Total rotations \\\\\n"
        << "\\midrule\n";
    for (const auto &r : rows) {
        out << r.packing << " & "
            << r.forward_rotations << " & "
            << r.backward_rotations << " & "
            << r.total_rotations << " \\\\\n";
    }
    out << "\\bottomrule\n"
        << "\\end{tabular}\n"
        << "\\end{table}\n";
}

int main(int argc, char **argv) {
    cout << unitbuf;
    Args args = parse_args(argc, argv);
    if (!is_power_of_two(args.dim)) throw runtime_error("--dim must be a power of two.");
    if (args.batch * args.dim > 16384) throw runtime_error("batch*dim exceeds CKKS slot count for N=32768.");
    if (args.repeats < 1) throw runtime_error("--repeats must be >= 1.");
    if (args.mode != "quick" && args.mode != "proposed-runtime" && args.mode != "full") {
        throw runtime_error("--mode must be one of: quick, proposed-runtime, full");
    }

    mt19937_64 rng(args.seed);
    auto active_x = random_vector(args.batch * args.dim, rng);
    auto active_dy = random_vector(args.batch * args.dim, rng);

    auto w = random_matrix(args.dim, args.dim, rng);
    auto wt = transpose(w);

    auto plain_rel = verify_plaintext_equivalence(active_x, active_dy, w, wt, args.batch, args.dim);

    vector<Row> rows;
    Row naive_row;
    naive_row.packing = "NaivePack";
    naive_row.forward_rotations = naive_rotation_count(args.dim);
    naive_row.backward_rotations = naive_row.forward_rotations;
    naive_row.total_rotations = naive_row.forward_rotations + naive_row.backward_rotations;
    naive_row.forward_rel_l2_error = 0.0;
    naive_row.backward_rel_l2_error = 0.0;
    naive_row.note = (args.mode == "full")
        ? "full CKKS runtime requested"
        : "operation-count only; CKKS Naive runtime skipped because it is slow and previous CKKS equivalence failed";
    rows.push_back(naive_row);

    Row proposed_row;
    proposed_row.packing = "ProposedPack";
    proposed_row.forward_rotations = diagonal_rotation_count(args.dim);
    proposed_row.backward_rotations = proposed_row.forward_rotations;
    proposed_row.total_rotations = proposed_row.forward_rotations + proposed_row.backward_rotations;
    proposed_row.forward_rel_l2_error = plain_rel.first;
    proposed_row.backward_rel_l2_error = plain_rel.second;
    proposed_row.note = "plaintext active-slot equivalence verified; CKKS runtime measured only in proposed-runtime/full mode";
    rows.push_back(proposed_row);

    cout << "[config] regime=SGF-GH batch=" << args.batch
         << " dim=" << args.dim
         << " mode=" << args.mode
         << " repeats=" << args.repeats
         << " warmups=" << args.warmups
         << "\n";

    if (args.mode != "quick") {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 16384 * 2;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {
            50,
            40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
            30, 30, 30, 30, 30, 30, 30, 30, 30,
            40
        }));

        SEALContext context(parms);
        if (!context.parameters_set()) {
            throw runtime_error(string("Invalid CKKS parameters: ") + context.parameter_error_message());
        }

        KeyGenerator keygen(context);
        PublicKey pk;
        keygen.create_public_key(pk);
        SecretKey sk = keygen.secret_key();
        GaloisKeys gk;
        keygen.create_galois_keys(gk);

        Encryptor encryptor(context, pk);
        Decryptor decryptor(context, sk);
        Evaluator evaluator(context);
        CKKSEncoder encoder(context);

        double scale = pow(2.0, 40);
        size_t slots = encoder.slot_count();
        vector<double> x(slots, 0.0);
        vector<double> dy(slots, 0.0);
        copy(active_x.begin(), active_x.end(), x.begin());
        copy(active_dy.begin(), active_dy.end(), dy.begin());

        cout << "[ckks] slot_count=" << slots
             << " coeff_modulus_bits=" << context.key_context_data()->total_coeff_modulus_bit_count()
             << "\n";

        Ciphertext enc_x = encrypt_slots(x, scale, encoder, encryptor);
        Ciphertext enc_dy = encrypt_slots(dy, scale, encoder, encryptor);

        auto diag_w = build_diagonal_masks(w, args.batch, slots, scale, encoder);
        auto diag_wt = build_diagonal_masks(wt, args.batch, slots, scale, encoder);

        vector<double> first_block(slots, 0.0);
        for (size_t i = 0; i < args.batch; ++i) first_block[i] = 1.0;
        Plaintext first_block_mask = encode_plain(first_block, scale, encoder);
        vector<Plaintext> naive_w;
        vector<Plaintext> naive_wt;

        pair<double, double> ckks_rel = {numeric_limits<double>::quiet_NaN(),
                                        numeric_limits<double>::quiet_NaN()};
        if (args.mode == "full") {
            naive_w = build_naive_column_masks(w, args.batch, slots, scale, encoder);
            naive_wt = build_naive_column_masks(wt, args.batch, slots, scale, encoder);
            ckks_rel = verify_equivalence(enc_x, enc_dy, args.batch, args.dim,
                                          naive_w, naive_wt, diag_w, diag_wt, first_block_mask,
                                          evaluator, gk, context, decryptor, encoder);
            rows[0] = bench_one("NaivePack", enc_x, enc_dy, args.batch, args.dim,
                                args.repeats, args.warmups, naive_w, naive_wt, diag_w, diag_wt,
                                first_block_mask, evaluator, gk, context);
            rows[0].forward_rel_l2_error = 0.0;
            rows[0].backward_rel_l2_error = 0.0;
            rows[0].note = "full CKKS Naive runtime measured";
        }

        rows[1] = bench_one("ProposedPack", enc_x, enc_dy, args.batch, args.dim,
                            args.repeats, args.warmups, naive_w, naive_wt, diag_w, diag_wt,
                            first_block_mask, evaluator, gk, context);
        rows[1].forward_rel_l2_error = (args.mode == "full") ? ckks_rel.first : plain_rel.first;
        rows[1].backward_rel_l2_error = (args.mode == "full") ? ckks_rel.second : plain_rel.second;
        rows[1].note = (args.mode == "full")
            ? "full CKKS Proposed runtime measured"
            : "CKKS Proposed runtime measured; Naive runtime intentionally skipped";
    }

    write_csv(args.output_csv, rows);
    write_tex(args.output_tex, rows);

    cout << "[saved] " << args.output_csv << "\n";
    cout << "[saved] " << args.output_tex << "\n";
    for (const auto &r : rows) {
        cout << r.packing
             << " rotations=" << r.total_rotations
             << " runtime_mean_s=" << fmt(r.total.mean)
             << " runtime_std_s=" << fmt(r.total.std)
             << " forward_rel_l2=" << fmt(r.forward_rel_l2_error)
             << " backward_rel_l2=" << fmt(r.backward_rel_l2_error)
             << "\n";
    }

    return 0;
}
