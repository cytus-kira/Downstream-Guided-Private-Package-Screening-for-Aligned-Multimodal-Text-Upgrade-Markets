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
#include <unordered_set>
#include <vector>

using namespace seal;
using Clock = std::chrono::high_resolution_clock;

struct Config {
    std::vector<int> rows{100, 500, 1000, 5000, 10000};
    std::vector<int> dims{16, 32, 64};
    std::vector<std::string> schemes{
        "baseline_random_noop",
        "baseline_cosine_ctpt",
        "baseline_coreset_distance_ctpt",
        "baseline_linear_student_ctpt",
        "ours_dcc_row_simd_ctpt",
        "ours_dcc_pkg_simd_ctpt",
        "ours_student_pkg_ctpt",
        "ours_student_row_ctpt",
        "ours_structural_ctpt",
        "ours_packaged_structural_ctpt",
        "ours_structural_ctct",
        "ours_packaged_structural_ctct",
        "baseline_poly2_fusion_ctct",
        "baseline_uncertainty_poly4_ctpt",
        "baseline_coreset_all_distances_ctpt",
        "baseline_badge_components_poly4_ctpt",
        "baseline_kmeans_all_distances_ctpt",
        "baseline_typiclust_sqrt_poly4_ctpt",
        "ours_krr_row_exp_poly4_ctpt",
        "ours_krr_pkg_exp_poly4_ctpt",
    };
    std::vector<int> coeff_modulus_bits{45, 32, 32, 32, 32, 45};
    std::string out_csv{"ckks_seal_results.csv"};
    std::size_t poly_modulus_degree = 8192;
    int scale_bits = 32;
    int package_size = 4;
    int student_summary_dim = 10;
    int repeats = 5;
    int warmups = 0;
    std::uint64_t seed = 42;
    bool validate = false;
    bool measure_decrypt_all = false;
    int threshold_parties = 1;
    int krr_landmarks = 1000;
    int coreset_references = 800;
    int kmeans_centers = 20;
    int typiclust_neighbors = 8;
};

struct Inputs {
    int logical_rows = 0;
    int scored_packages = 0;
    int raw_dim = 0;
    int student_summary_dim = 0;
    double img_prepare_encrypt_ms = 0.0;
    double weak_prepare_encrypt_ms = 0.0;
    double cand_prepare_encrypt_ms = 0.0;
    double dcc_row_prepare_encrypt_ms = 0.0;
    double dcc_package_prepare_encrypt_ms = 0.0;
    double student_package_prepare_encrypt_ms = 0.0;
    double package_feature_prepare_encrypt_ms = 0.0;
    std::vector<std::vector<Ciphertext>> img_ct;
    std::vector<std::vector<Ciphertext>> weak_ct;
    std::vector<std::vector<Ciphertext>> cand_ct;
    std::vector<std::vector<Plaintext>> img_pt;
    std::vector<std::vector<Plaintext>> weak_pt;
    std::vector<std::vector<Plaintext>> cand_pt;
    std::vector<Plaintext> scalar_weight_pt;
    std::vector<std::vector<Ciphertext>> dcc_row_summary_ct;
    std::vector<std::vector<Ciphertext>> dcc_package_summary_ct;
    std::vector<Plaintext> dcc_weight_pt;
    double dcc_intercept = 0.0;
    std::vector<std::vector<Ciphertext>> package_summary_ct;
    std::vector<std::vector<Ciphertext>> package_feature_ct;
    std::vector<Ciphertext> package_norm_term_ct;
    std::vector<Plaintext> student_weight_pt;
    double student_intercept = 0.03;
};

struct Stats {
    std::uint64_t ct_ct_mults = 0;
    std::uint64_t ct_pt_mults = 0;
    std::uint64_t rotations = 0;
    std::uint64_t additions = 0;
    std::uint64_t relinearizations = 0;
    std::uint64_t rescales = 0;
};

static std::vector<std::string> split(const std::string &text, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, sep)) {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

static std::vector<int> parse_int_list(const std::string &text)
{
    std::vector<int> out;
    for (const auto &item : split(text, ',')) {
        out.push_back(std::stoi(item));
    }
    return out;
}

static std::vector<std::string> default_schemes()
{
    return {
        "baseline_random_noop",
        "baseline_cosine_ctpt",
        "baseline_coreset_distance_ctpt",
        "baseline_linear_student_ctpt",
        "ours_dcc_row_simd_ctpt",
        "ours_dcc_pkg_simd_ctpt",
        "ours_student_pkg_ctpt",
        "ours_student_row_ctpt",
        "ours_structural_ctpt",
        "ours_packaged_structural_ctpt",
        "ours_structural_ctct",
        "ours_packaged_structural_ctct",
        "baseline_poly2_fusion_ctct",
        "baseline_uncertainty_poly4_ctpt",
        "baseline_coreset_all_distances_ctpt",
        "baseline_badge_components_poly4_ctpt",
        "baseline_kmeans_all_distances_ctpt",
        "baseline_typiclust_sqrt_poly4_ctpt",
        "ours_krr_row_exp_poly4_ctpt",
        "ours_krr_pkg_exp_poly4_ctpt",
    };
}

static void print_help()
{
    std::cout
        << "CKKS SEAL benchmark for encrypted middle-layer scoring.\n\n"
        << "Options:\n"
        << "  --rows 100,500,1000,5000,10000\n"
        << "  --dims 16,32,64\n"
        << "  --schemes all OR comma list\n"
        << "  --repeats 5\n"
        << "  --warmups 0\n"
        << "  --poly 8192\n"
        << "  --coeff-bits 45,32,32,32,32,45\n"
        << "  --scale-bits 32\n"
        << "  --package-size 2\n"
        << "  --student-summary-dim 10\n"
        << "  --seed 42\n"
        << "  --out outputs/ckks_seal_bench/ckks_seal_results.csv\n"
        << "  --measure-decrypt-all\n"
        << "  --threshold-parties 1\n"
        << "  --krr-landmarks 1000\n"
        << "  --coreset-references 800\n"
        << "  --kmeans-centers 20\n"
        << "  --typiclust-neighbors 8\n"
        << "  --validate\n";
}

static Config parse_args(int argc, char **argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };
        if (key == "--help" || key == "-h") {
            print_help();
            std::exit(0);
        } else if (key == "--rows") {
            cfg.rows = parse_int_list(need_value(key));
        } else if (key == "--dims") {
            cfg.dims = parse_int_list(need_value(key));
        } else if (key == "--schemes") {
            std::string text = need_value(key);
            cfg.schemes = (text == "all") ? default_schemes() : split(text, ',');
        } else if (key == "--repeats") {
            cfg.repeats = std::stoi(need_value(key));
        } else if (key == "--warmups") {
            cfg.warmups = std::stoi(need_value(key));
        } else if (key == "--poly") {
            cfg.poly_modulus_degree = static_cast<std::size_t>(std::stoul(need_value(key)));
        } else if (key == "--coeff-bits") {
            cfg.coeff_modulus_bits = parse_int_list(need_value(key));
        } else if (key == "--scale-bits") {
            cfg.scale_bits = std::stoi(need_value(key));
        } else if (key == "--package-size") {
            cfg.package_size = std::stoi(need_value(key));
        } else if (key == "--student-summary-dim") {
            cfg.student_summary_dim = std::stoi(need_value(key));
        } else if (key == "--seed") {
            cfg.seed = static_cast<std::uint64_t>(std::stoull(need_value(key)));
        } else if (key == "--out") {
            cfg.out_csv = need_value(key);
        } else if (key == "--measure-decrypt-all") {
            cfg.measure_decrypt_all = true;
        } else if (key == "--threshold-parties") {
            cfg.threshold_parties = std::stoi(need_value(key));
        } else if (key == "--krr-landmarks") {
            cfg.krr_landmarks = std::stoi(need_value(key));
        } else if (key == "--coreset-references") {
            cfg.coreset_references = std::stoi(need_value(key));
        } else if (key == "--kmeans-centers") {
            cfg.kmeans_centers = std::stoi(need_value(key));
        } else if (key == "--typiclust-neighbors") {
            cfg.typiclust_neighbors = std::stoi(need_value(key));
        } else if (key == "--validate") {
            cfg.validate = true;
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (cfg.rows.empty() || cfg.dims.empty() || cfg.schemes.empty()) {
        throw std::runtime_error("rows, dims, and schemes must be non-empty.");
    }
    if (cfg.repeats <= 0 || cfg.warmups < 0 || cfg.package_size <= 0 ||
        cfg.student_summary_dim <= 0 || cfg.scale_bits <= 0 || cfg.threshold_parties <= 0 ||
        cfg.krr_landmarks <= 0 || cfg.coreset_references <= 0 ||
        cfg.kmeans_centers <= 0 || cfg.typiclust_neighbors <= 0) {
        throw std::runtime_error("repeats, warmups, package-size, student-summary-dim, and scale-bits must be valid.");
    }
    return cfg;
}

static std::string join_ints(const std::vector<int> &xs)
{
    std::ostringstream os;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i) {
            os << "-";
        }
        os << xs[i];
    }
    return os.str();
}

static std::vector<double> random_slots(
    std::size_t active,
    std::size_t slots,
    std::mt19937_64 &rng,
    std::uniform_real_distribution<double> &dist)
{
    std::vector<double> values(slots, 0.0);
    for (std::size_t i = 0; i < active; ++i) {
        values[i] = dist(rng);
    }
    return values;
}

static std::size_t ciphertext_bytes(const Ciphertext &ct)
{
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ct.save(ss);
    return static_cast<std::size_t>(ss.tellp());
}

static void force_scale(Ciphertext &ct, double scale)
{
    ct.scale() = scale;
}

static Ciphertext multiply_scalar(
    const Ciphertext &src,
    double value,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    Plaintext scalar;
    encoder.encode(value, src.parms_id(), scale, scalar);
    Ciphertext out;
    evaluator.multiply_plain(src, scalar, out);
    stats.ct_pt_mults++;
    evaluator.rescale_to_next_inplace(out);
    stats.rescales++;
    force_scale(out, scale);
    return out;
}

static Ciphertext dot_ctpt(
    const std::vector<Ciphertext> &lhs,
    const std::vector<Plaintext> &rhs,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    Ciphertext acc;
    bool has_acc = false;
    for (std::size_t d = 0; d < lhs.size(); ++d) {
        Ciphertext tmp;
        evaluator.multiply_plain(lhs[d], rhs[d], tmp);
        stats.ct_pt_mults++;
        evaluator.rescale_to_next_inplace(tmp);
        stats.rescales++;
        force_scale(tmp, scale);
        if (!has_acc) {
            acc = std::move(tmp);
            has_acc = true;
        } else {
            evaluator.add_inplace(acc, tmp);
            stats.additions++;
        }
    }
    return acc;
}

static Ciphertext dot_ctct(
    const std::vector<Ciphertext> &lhs,
    const std::vector<Ciphertext> &rhs,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    double scale,
    Stats &stats)
{
    Ciphertext acc;
    bool has_acc = false;
    for (std::size_t d = 0; d < lhs.size(); ++d) {
        Ciphertext tmp;
        evaluator.multiply(lhs[d], rhs[d], tmp);
        stats.ct_ct_mults++;
        evaluator.relinearize_inplace(tmp, relin_keys);
        stats.relinearizations++;
        evaluator.rescale_to_next_inplace(tmp);
        stats.rescales++;
        force_scale(tmp, scale);
        if (!has_acc) {
            acc = std::move(tmp);
            has_acc = true;
        } else {
            evaluator.add_inplace(acc, tmp);
            stats.additions++;
        }
    }
    return acc;
}

static Ciphertext coreset_distance_ctpt(
    const std::vector<Ciphertext> &cand,
    const std::vector<Plaintext> &center,
    const std::vector<Plaintext> &constant_weights,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    // Coverage/CoreSet-style scoring can be reduced to a distance to one public
    // center when center membership is fixed outside the encrypted middle path.
    // For normalized features, ||center||^2 is constant and ||cand||^2 is often
    // cached; here we count the encrypted evaluator work for the CT-PT part
    // plus a lightweight encrypted linear norm proxy so the scheme has its own
    // timing footprint rather than aliasing cosine exactly.
    Ciphertext similarity = dot_ctpt(cand, center, evaluator, scale, stats);
    Ciphertext norm_proxy = dot_ctpt(cand, constant_weights, evaluator, scale, stats);
    evaluator.add_inplace(similarity, norm_proxy);
    stats.additions++;
    force_scale(similarity, scale);
    return similarity;
}

static Ciphertext combine_structural(
    const Ciphertext &img_cand,
    const Ciphertext &img_weak,
    const Ciphertext &cand_weak,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    // Matches the paper-side structural form used by the selection code:
    // -cos(img,cand) - 0.10*cos(cand,weak) - 0.10*(cos(img,cand)-cos(img,weak)).
    // This is equivalent to -1.10*IC + 0.10*IW - 0.10*CW.
    Ciphertext out = multiply_scalar(img_cand, -1.10, encoder, evaluator, scale, stats);
    Ciphertext iw = multiply_scalar(img_weak, 0.10, encoder, evaluator, scale, stats);
    Ciphertext cw = multiply_scalar(cand_weak, -0.10, encoder, evaluator, scale, stats);
    evaluator.add_inplace(out, iw);
    evaluator.add_inplace(out, cw);
    stats.additions += 2;
    force_scale(out, scale);
    return out;
}

static Ciphertext package_mean(
    const Ciphertext &score,
    int package_size,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const GaloisKeys &galois_keys,
    double scale,
    Stats &stats)
{
    Ciphertext out = score;
    for (int shift = 1; shift < package_size; ++shift) {
        Ciphertext rotated;
        evaluator.rotate_vector(score, -shift, galois_keys, rotated);
        stats.rotations++;
        evaluator.add_inplace(out, rotated);
        stats.additions++;
    }
    // The ranking is unchanged by dividing all fixed-size packages by the same
    // constant, so normalization is deferred outside the encrypted middle path.
    (void)encoder;
    (void)evaluator;
    (void)scale;
    return out;
}

static Ciphertext add_plain_scalar(
    Ciphertext src,
    double value,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    Plaintext scalar;
    encoder.encode(value, src.parms_id(), scale, scalar);
    evaluator.add_plain_inplace(src, scalar);
    stats.additions++;
    force_scale(src, scale);
    return src;
}

static std::vector<double> deterministic_unit_reference(int index, int dim)
{
    std::vector<double> ref(static_cast<std::size_t>(dim), 0.0);
    double norm2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double value =
            std::sin(0.017 * static_cast<double>((index + 1) * (d + 1))) +
            0.5 * std::cos(0.031 * static_cast<double>((index + 3) * (d + 1)));
        ref[static_cast<std::size_t>(d)] = value;
        norm2 += value * value;
    }
    const double inv_norm = 1.0 / std::sqrt(std::max(norm2, 1e-18));
    for (double &value : ref) {
        value *= inv_norm;
    }
    return ref;
}

static Ciphertext dot_dynamic_ctpt(
    const std::vector<Ciphertext> &lhs,
    const std::vector<double> &rhs,
    double rhs_scale,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    Ciphertext acc;
    bool has_acc = false;
    for (std::size_t d = 0; d < lhs.size(); ++d) {
        Plaintext coeff;
        encoder.encode(rhs[d] * rhs_scale, lhs[d].parms_id(), scale, coeff);
        Ciphertext tmp;
        evaluator.multiply_plain(lhs[d], coeff, tmp);
        stats.ct_pt_mults++;
        evaluator.rescale_to_next_inplace(tmp);
        stats.rescales++;
        force_scale(tmp, scale);
        if (!has_acc) {
            acc = std::move(tmp);
            has_acc = true;
        } else {
            evaluator.add_inplace(acc, tmp);
            stats.additions++;
        }
    }
    return acc;
}

static Ciphertext multiply_ctct_rescale(
    const Ciphertext &lhs,
    const Ciphertext &rhs,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    double scale,
    Stats &stats)
{
    Ciphertext out;
    evaluator.multiply(lhs, rhs, out);
    stats.ct_ct_mults++;
    evaluator.relinearize_inplace(out, relin_keys);
    stats.relinearizations++;
    evaluator.rescale_to_next_inplace(out);
    stats.rescales++;
    force_scale(out, scale);
    return out;
}

static Ciphertext multiply_plain_rescale(
    const Ciphertext &src,
    double coefficient,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    Plaintext coeff;
    encoder.encode(coefficient, src.parms_id(), scale, coeff);
    Ciphertext out;
    evaluator.multiply_plain(src, coeff, out);
    stats.ct_pt_mults++;
    evaluator.rescale_to_next_inplace(out);
    stats.rescales++;
    force_scale(out, scale);
    return out;
}

static Ciphertext polynomial4_two_level(
    const Ciphertext &x,
    const std::vector<double> &coeff,
    double weight,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    double scale,
    Stats &stats)
{
    if (coeff.size() != 5) {
        throw std::runtime_error("Degree-4 polynomial requires five coefficients.");
    }
    Ciphertext x2 = multiply_ctct_rescale(x, x, evaluator, relin_keys, scale, stats);
    Ciphertext x4 = multiply_ctct_rescale(x2, x2, evaluator, relin_keys, scale, stats);
    std::vector<Ciphertext> terms;
    if (std::abs(weight * coeff[1]) > 1e-15) {
        terms.push_back(multiply_plain_rescale(
            x, weight * coeff[1], encoder, evaluator, scale, stats));
    }
    if (std::abs(weight * coeff[2]) > 1e-15) {
        terms.push_back(multiply_plain_rescale(
            x2, weight * coeff[2], encoder, evaluator, scale, stats));
    }
    if (std::abs(weight * coeff[3]) > 1e-15) {
        Ciphertext x_at_x2 = x;
        evaluator.mod_switch_to_inplace(x_at_x2, x2.parms_id());
        force_scale(x_at_x2, scale);
        Ciphertext x3 = multiply_ctct_rescale(
            x_at_x2, x2, evaluator, relin_keys, scale, stats);
        terms.push_back(multiply_plain_rescale(
            x3, weight * coeff[3], encoder, evaluator, scale, stats));
    }
    if (std::abs(weight * coeff[4]) > 1e-15) {
        terms.push_back(multiply_plain_rescale(
            x4, weight * coeff[4], encoder, evaluator, scale, stats));
    }
    if (terms.empty()) {
        throw std::runtime_error("Polynomial has no non-constant encrypted terms.");
    }

    const auto target = terms.back().parms_id();
    for (auto &term : terms) {
        if (term.parms_id() != target) {
            evaluator.mod_switch_to_inplace(term, target);
        }
        force_scale(term, scale);
    }
    Ciphertext result = std::move(terms.front());
    for (std::size_t i = 1; i < terms.size(); ++i) {
        evaluator.add_inplace(result, terms[i]);
        stats.additions++;
    }

    Plaintext constant;
    encoder.encode(weight * coeff[0], target, scale, constant);
    evaluator.add_plain_inplace(result, constant);
    stats.additions++;
    force_scale(result, scale);
    return result;
}

static Ciphertext normalized_squared_distance(
    const std::vector<Ciphertext> &candidate,
    const std::vector<double> &reference,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    Ciphertext distance = dot_dynamic_ctpt(
        candidate, reference, -2.0, encoder, evaluator, scale, stats);
    return add_plain_scalar(distance, 2.0, encoder, evaluator, scale, stats);
}

static Ciphertext krr_poly4_score_chunk(
    const std::vector<Ciphertext> &candidate,
    const Ciphertext *encrypted_negative_half_norm,
    int landmark_count,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    double scale,
    Stats &stats)
{
    // Unit-normalized rows and their package means have norm at most one, so
    // d^2 is in [0,4]. The registered sigma^2 floor 0.5 gives u in [-4,0].
    constexpr double sigma2 = 0.5;
    const std::vector<double> exp_poly{
        0.9963358096138180,
        0.9534382874063090,
        0.3987763197687612,
        0.0819512647685308,
        0.0066488054692884,
    };
    Ciphertext score;
    bool has_score = false;
    for (int r = 0; r < landmark_count; ++r) {
        const auto landmark = deterministic_unit_reference(r, static_cast<int>(candidate.size()));
        Ciphertext u = dot_dynamic_ctpt(
            candidate, landmark, 1.0 / sigma2, encoder, evaluator, scale, stats);
        if (encrypted_negative_half_norm != nullptr) {
            Ciphertext norm_term = *encrypted_negative_half_norm;
            evaluator.mod_switch_to_inplace(norm_term, u.parms_id());
            force_scale(norm_term, scale);
            evaluator.add_inplace(u, norm_term);
            stats.additions++;
            u = add_plain_scalar(
                u, -0.5 / sigma2, encoder, evaluator, scale, stats);
        } else {
            u = add_plain_scalar(u, -1.0 / sigma2, encoder, evaluator, scale, stats);
        }
        const double alpha =
            std::sin(0.013 * static_cast<double>(r + 1)) /
            std::sqrt(static_cast<double>(landmark_count));
        Ciphertext term = polynomial4_two_level(
            u, exp_poly, alpha, encoder, evaluator, relin_keys, scale, stats);
        if (!has_score) {
            score = std::move(term);
            has_score = true;
        } else {
            evaluator.add_inplace(score, term);
            stats.additions++;
        }
    }
    return score;
}

static Ciphertext uncertainty_poly4_score_chunk(
    const std::vector<Ciphertext> &candidate,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    double scale,
    Stats &stats)
{
    // The public linear logit head is normalized to [-3,3]. This polynomial
    // approximates sigmoid(z)(1-sigmoid(z)) on that interval.
    const auto head = deterministic_unit_reference(1000003, static_cast<int>(candidate.size()));
    Ciphertext logit = dot_dynamic_ctpt(
        candidate, head, 3.0, encoder, evaluator, scale, stats);
    const std::vector<double> uncertainty_poly{
        0.25,
        0.0,
        -0.0500068676,
        0.0,
        0.00314823380,
    };
    return polynomial4_two_level(
        logit, uncertainty_poly, 1.0, encoder, evaluator, relin_keys, scale, stats);
}

static Ciphertext downstream_student_residual_score(
    const std::vector<Ciphertext> &summary,
    const std::vector<Plaintext> &weights,
    double intercept,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    // Latest deployed scorer:
    //   Score(P) = z(s_top(P)) + kappa * beta^T normalize(h_P) + b.
    // The seller submits the package summary h_P after local packaging. The
    // buyer/platform coefficients, including z-score scaling and kappa, are
    // folded into plaintext weights. Feature 0 is the residual structural
    // anchor coordinate, so the encrypted online path is only a CT-PT linear
    // score plus a plaintext intercept.
    Ciphertext score = dot_ctpt(summary, weights, evaluator, scale, stats);
    Plaintext intercept_pt;
    encoder.encode(intercept, score.parms_id(), scale, intercept_pt);
    evaluator.add_plain_inplace(score, intercept_pt);
    stats.additions++;
    force_scale(score, scale);
    return score;
}

static Ciphertext dcc_simd_linear_score(
    const std::vector<Ciphertext> &summary,
    const std::vector<Plaintext> &weights,
    double intercept,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    double scale,
    Stats &stats)
{
    // Downstream-Calibrated Coverage (DCC) online path:
    //
    //   Score = beta_cov * cov
    //         + beta_label * label_task_margin
    //         + beta_struct * structural_prior
    //         + beta_corr * calibrated_task_correction
    //         + b.
    //
    // Each CKKS slot stores one row/package candidate.  The buyer/task-specific
    // beta values are plaintext constants learned offline from the downstream
    // calibration stage.  The encrypted online evaluator therefore performs a
    // single SIMD CT-PT dot product and one plaintext intercept addition.
    Ciphertext score = dot_ctpt(summary, weights, evaluator, scale, stats);
    Plaintext intercept_pt;
    encoder.encode(intercept, score.parms_id(), scale, intercept_pt);
    evaluator.add_plain_inplace(score, intercept_pt);
    stats.additions++;
    force_scale(score, scale);
    return score;
}

static Inputs prepare_inputs(
    int logical_rows,
    int dim,
    int student_summary_dim,
    int package_size,
    std::size_t slots,
    CKKSEncoder &encoder,
    Encryptor &encryptor,
    double scale,
    std::mt19937_64 &rng)
{
    const std::size_t chunks = static_cast<std::size_t>((logical_rows + static_cast<int>(slots) - 1) / static_cast<int>(slots));
    const int scored_packages = (logical_rows + package_size - 1) / package_size;
    const std::size_t package_chunks =
        static_cast<std::size_t>((scored_packages + static_cast<int>(slots) - 1) / static_cast<int>(slots));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Inputs in;
    in.logical_rows = logical_rows;
    in.scored_packages = scored_packages;
    in.raw_dim = dim;
    in.student_summary_dim = student_summary_dim;
    in.img_ct.resize(chunks);
    in.weak_ct.resize(chunks);
    in.cand_ct.resize(chunks);
    in.img_pt.resize(chunks);
    in.weak_pt.resize(chunks);
    in.cand_pt.resize(chunks);

    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t start = c * slots;
        const std::size_t active = std::min<std::size_t>(slots, static_cast<std::size_t>(logical_rows) - start);
        in.img_ct[c].resize(dim);
        in.weak_ct[c].resize(dim);
        in.cand_ct[c].resize(dim);
        in.img_pt[c].resize(dim);
        in.weak_pt[c].resize(dim);
        in.cand_pt[c].resize(dim);
        for (int d = 0; d < dim; ++d) {
            auto img = random_slots(active, slots, rng, dist);
            auto weak = random_slots(active, slots, rng, dist);
            auto cand = random_slots(active, slots, rng, dist);
            Plaintext img_plain, weak_plain, cand_plain;
            auto img_start = Clock::now();
            encoder.encode(img, scale, img_plain);
            encryptor.encrypt(img_plain, in.img_ct[c][d]);
            auto img_end = Clock::now();
            in.img_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(img_end - img_start).count();
            auto weak_start = Clock::now();
            encoder.encode(weak, scale, weak_plain);
            encryptor.encrypt(weak_plain, in.weak_ct[c][d]);
            auto weak_end = Clock::now();
            in.weak_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(weak_end - weak_start).count();
            auto cand_start = Clock::now();
            encoder.encode(cand, scale, cand_plain);
            encryptor.encrypt(cand_plain, in.cand_ct[c][d]);
            auto cand_end = Clock::now();
            in.cand_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(cand_end - cand_start).count();
            in.img_pt[c][d] = std::move(img_plain);
            in.weak_pt[c][d] = std::move(weak_plain);
            in.cand_pt[c][d] = std::move(cand_plain);
        }
    }

    in.scalar_weight_pt.resize(dim);
    for (int d = 0; d < dim; ++d) {
        const double w = 0.25 * std::sin(static_cast<double>(d + 1));
        std::vector<double> weights(slots, w);
        encoder.encode(weights, scale, in.scalar_weight_pt[d]);
    }

    const int dcc_summary_dim = 4;
    in.dcc_row_summary_ct.resize(chunks);
    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t start = c * slots;
        const std::size_t active = std::min<std::size_t>(slots, static_cast<std::size_t>(logical_rows) - start);
        in.dcc_row_summary_ct[c].resize(dcc_summary_dim);
        for (int d = 0; d < dcc_summary_dim; ++d) {
            auto feature = random_slots(active, slots, rng, dist);
            Plaintext feature_plain;
            auto start = Clock::now();
            encoder.encode(feature, scale, feature_plain);
            encryptor.encrypt(feature_plain, in.dcc_row_summary_ct[c][d]);
            auto end = Clock::now();
            in.dcc_row_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    in.dcc_package_summary_ct.resize(package_chunks);
    for (std::size_t c = 0; c < package_chunks; ++c) {
        const std::size_t start = c * slots;
        const std::size_t active =
            std::min<std::size_t>(slots, static_cast<std::size_t>(scored_packages) - start);
        in.dcc_package_summary_ct[c].resize(dcc_summary_dim);
        for (int d = 0; d < dcc_summary_dim; ++d) {
            auto feature = random_slots(active, slots, rng, dist);
            Plaintext feature_plain;
            auto start = Clock::now();
            encoder.encode(feature, scale, feature_plain);
            encryptor.encrypt(feature_plain, in.dcc_package_summary_ct[c][d]);
            auto end = Clock::now();
            in.dcc_package_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    // Representative DCC plaintext coefficients.  In the real pipeline these
    // are buyer/task calibrated and may change by dataset; CKKS cost is
    // independent of the numeric values as long as the scorer remains linear.
    const std::vector<double> dcc_weights{1.0, 0.25, 0.25, 0.10};
    in.dcc_weight_pt.resize(dcc_summary_dim);
    for (int d = 0; d < dcc_summary_dim; ++d) {
        std::vector<double> weights(slots, dcc_weights[static_cast<std::size_t>(d)]);
        encoder.encode(weights, scale, in.dcc_weight_pt[d]);
    }
    in.dcc_intercept = 0.0;

    in.package_summary_ct.resize(package_chunks);
    for (std::size_t c = 0; c < package_chunks; ++c) {
        const std::size_t start = c * slots;
        const std::size_t active =
            std::min<std::size_t>(slots, static_cast<std::size_t>(scored_packages) - start);
        in.package_summary_ct[c].resize(student_summary_dim);
        for (int d = 0; d < student_summary_dim; ++d) {
            auto feature = random_slots(active, slots, rng, dist);
            Plaintext feature_plain;
            auto start = Clock::now();
            encoder.encode(feature, scale, feature_plain);
            encryptor.encrypt(feature_plain, in.package_summary_ct[c][d]);
            auto end = Clock::now();
            in.student_package_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
    in.package_feature_ct.resize(package_chunks);
    in.package_norm_term_ct.resize(package_chunks);
    for (std::size_t c = 0; c < package_chunks; ++c) {
        const std::size_t start = c * slots;
        const std::size_t active =
            std::min<std::size_t>(slots, static_cast<std::size_t>(scored_packages) - start);
        in.package_feature_ct[c].resize(dim);
        std::uniform_real_distribution<double> feature_dist(
            -1.0 / std::sqrt(static_cast<double>(dim)),
            1.0 / std::sqrt(static_cast<double>(dim)));
        for (int d = 0; d < dim; ++d) {
            auto feature = random_slots(active, slots, rng, feature_dist);
            Plaintext feature_plain;
            auto start_time = Clock::now();
            encoder.encode(feature, scale, feature_plain);
            encryptor.encrypt(feature_plain, in.package_feature_ct[c][d]);
            auto end_time = Clock::now();
            in.package_feature_prepare_encrypt_ms +=
                std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }
        std::vector<double> negative_half_norm(slots, 0.0);
        for (std::size_t i = 0; i < active; ++i) {
            // Representative package-mean norm term. The real seller computes
            // -||x||^2/(2 sigma^2) exactly from its local package summary.
            negative_half_norm[i] = -1.0 / 3.0;
        }
        Plaintext norm_plain;
        auto norm_start = Clock::now();
        encoder.encode(negative_half_norm, scale, norm_plain);
        encryptor.encrypt(norm_plain, in.package_norm_term_ct[c]);
        auto norm_end = Clock::now();
        in.package_feature_prepare_encrypt_ms +=
            std::chrono::duration<double, std::milli>(norm_end - norm_start).count();
    }
    in.student_weight_pt.resize(student_summary_dim);
    for (int d = 0; d < student_summary_dim; ++d) {
        double w = 0.10 * std::cos(static_cast<double>(d + 1));
        if (d == 0) {
            // Structural residual anchor plus learned correction folded into
            // the first package-summary coordinate.
            w += 1.0;
        }
        std::vector<double> weights(slots, w);
        encoder.encode(weights, scale, in.student_weight_pt[d]);
    }
    in.student_intercept = 0.03;
    return in;
}

static std::vector<Ciphertext> run_scheme(
    const std::string &scheme,
    const Inputs &in,
    int package_size,
    int krr_landmarks,
    int coreset_references,
    int kmeans_centers,
    int typiclust_neighbors,
    CKKSEncoder &encoder,
    Evaluator &evaluator,
    const RelinKeys &relin_keys,
    const GaloisKeys &galois_keys,
    double scale,
    Stats &stats)
{
    std::vector<Ciphertext> outputs;
    outputs.reserve(in.img_ct.size());

    if (scheme == "ours_krr_pkg_exp_poly4_ctpt") {
        outputs.reserve(in.package_feature_ct.size());
        for (const auto &chunk : in.package_feature_ct) {
            outputs.push_back(krr_poly4_score_chunk(
                chunk, &in.package_norm_term_ct[outputs.size()], krr_landmarks,
                encoder, evaluator, relin_keys, scale, stats));
        }
        return outputs;
    }
    if (scheme == "ours_student_pkg_ctpt") {
        outputs.reserve(in.package_summary_ct.size());
        for (std::size_t c = 0; c < in.package_summary_ct.size(); ++c) {
            outputs.push_back(downstream_student_residual_score(
                in.package_summary_ct[c], in.student_weight_pt, in.student_intercept,
                encoder, evaluator, scale, stats));
        }
        return outputs;
    }
    if (scheme == "ours_dcc_pkg_simd_ctpt") {
        outputs.reserve(in.dcc_package_summary_ct.size());
        for (std::size_t c = 0; c < in.dcc_package_summary_ct.size(); ++c) {
            outputs.push_back(dcc_simd_linear_score(
                in.dcc_package_summary_ct[c], in.dcc_weight_pt, in.dcc_intercept,
                encoder, evaluator, scale, stats));
        }
        return outputs;
    }

    for (std::size_t c = 0; c < in.img_ct.size(); ++c) {
        if (scheme == "baseline_random_noop") {
            // Random purchase performs no encrypted scoring in the online
            // middle path. Returning no ciphertexts intentionally records the
            // near-zero evaluator cost for this acquisition rule.
            continue;
        } else if (scheme == "baseline_cosine_ctpt") {
            outputs.push_back(dot_ctpt(in.img_ct[c], in.cand_pt[c], evaluator, scale, stats));
        } else if (scheme == "baseline_uncertainty_poly4_ctpt") {
            outputs.push_back(uncertainty_poly4_score_chunk(
                in.img_ct[c], encoder, evaluator, relin_keys, scale, stats));
        } else if (scheme == "baseline_coreset_all_distances_ctpt" ||
                   scheme == "baseline_kmeans_all_distances_ctpt") {
            const int references =
                scheme == "baseline_coreset_all_distances_ctpt"
                    ? coreset_references
                    : kmeans_centers;
            for (int r = 0; r < references; ++r) {
                const auto reference =
                    deterministic_unit_reference(2000000 + r, static_cast<int>(in.img_ct[c].size()));
                outputs.push_back(normalized_squared_distance(
                    in.img_ct[c], reference, encoder, evaluator, scale, stats));
            }
        } else if (scheme == "baseline_badge_components_poly4_ctpt") {
            outputs.push_back(uncertainty_poly4_score_chunk(
                in.img_ct[c], encoder, evaluator, relin_keys, scale, stats));
            const std::vector<double> sqrt_poly{
                0.22988499,
                1.10318203,
                -0.41972626,
                0.10074545,
                -0.00929105,
            };
            for (int r = 0; r < coreset_references; ++r) {
                const auto reference =
                    deterministic_unit_reference(3000000 + r, static_cast<int>(in.img_ct[c].size()));
                Ciphertext distance = normalized_squared_distance(
                    in.img_ct[c], reference, encoder, evaluator, scale, stats);
                outputs.push_back(polynomial4_two_level(
                    distance, sqrt_poly, 1.0, encoder, evaluator, relin_keys, scale, stats));
            }
        } else if (scheme == "baseline_typiclust_sqrt_poly4_ctpt") {
            const std::vector<double> sqrt_poly{
                0.22988499,
                1.10318203,
                -0.41972626,
                0.10074545,
                -0.00929105,
            };
            Ciphertext density;
            bool has_density = false;
            for (int r = 0; r < typiclust_neighbors; ++r) {
                const auto reference =
                    deterministic_unit_reference(4000000 + r, static_cast<int>(in.img_ct[c].size()));
                Ciphertext distance = normalized_squared_distance(
                    in.img_ct[c], reference, encoder, evaluator, scale, stats);
                Ciphertext root = polynomial4_two_level(
                    distance, sqrt_poly, -1.0 / static_cast<double>(typiclust_neighbors),
                    encoder, evaluator, relin_keys, scale, stats);
                if (!has_density) {
                    density = std::move(root);
                    has_density = true;
                } else {
                    evaluator.add_inplace(density, root);
                    stats.additions++;
                }
            }
            outputs.push_back(std::move(density));
        } else if (scheme == "ours_krr_row_exp_poly4_ctpt") {
            outputs.push_back(krr_poly4_score_chunk(
                in.img_ct[c], nullptr, krr_landmarks,
                encoder, evaluator, relin_keys, scale, stats));
        } else if (scheme == "baseline_coreset_distance_ctpt") {
            outputs.push_back(coreset_distance_ctpt(
                in.cand_ct[c], in.img_pt[c], in.scalar_weight_pt, evaluator, scale, stats));
        } else if (scheme == "baseline_linear_student_ctpt") {
            outputs.push_back(dot_ctpt(in.img_ct[c], in.scalar_weight_pt, evaluator, scale, stats));
        } else if (scheme == "ours_dcc_row_simd_ctpt") {
            outputs.push_back(dcc_simd_linear_score(
                in.dcc_row_summary_ct[c], in.dcc_weight_pt, in.dcc_intercept,
                encoder, evaluator, scale, stats));
        } else if (scheme == "ours_student_row_ctpt") {
            // A raw-row reference for the same residual-linear student idea.
            // This scores every raw row with a CT-PT linear form over the raw
            // feature vector, so it is intentionally more expensive than the
            // package-summary scorer.
            Ciphertext score = dot_ctpt(in.img_ct[c], in.scalar_weight_pt, evaluator, scale, stats);
            score = add_plain_scalar(score, 0.03, encoder, evaluator, scale, stats);
            outputs.push_back(std::move(score));
        } else if (scheme == "ours_structural_ctpt" || scheme == "ours_packaged_structural_ctpt") {
            Ciphertext ic = dot_ctpt(in.img_ct[c], in.cand_pt[c], evaluator, scale, stats);
            Ciphertext iw = dot_ctpt(in.img_ct[c], in.weak_pt[c], evaluator, scale, stats);
            Ciphertext cw = dot_ctpt(in.cand_ct[c], in.weak_pt[c], evaluator, scale, stats);
            Ciphertext score = combine_structural(ic, iw, cw, encoder, evaluator, scale, stats);
            if (scheme == "ours_packaged_structural_ctpt") {
                score = package_mean(score, package_size, encoder, evaluator, galois_keys, scale, stats);
            }
            outputs.push_back(std::move(score));
        } else if (scheme == "ours_structural_ctct" || scheme == "ours_packaged_structural_ctct" ||
                   scheme == "baseline_poly2_fusion_ctct") {
            Ciphertext ic = dot_ctct(in.img_ct[c], in.cand_ct[c], evaluator, relin_keys, scale, stats);
            Ciphertext iw = dot_ctct(in.img_ct[c], in.weak_ct[c], evaluator, relin_keys, scale, stats);
            Ciphertext cw = dot_ctct(in.cand_ct[c], in.weak_ct[c], evaluator, relin_keys, scale, stats);
            if (scheme == "baseline_poly2_fusion_ctct") {
                (void)iw;
                (void)cw;
                Ciphertext squared;
                evaluator.square(ic, squared);
                stats.ct_ct_mults++;
                evaluator.relinearize_inplace(squared, relin_keys);
                stats.relinearizations++;
                evaluator.rescale_to_next_inplace(squared);
                stats.rescales++;
                force_scale(squared, scale);
                outputs.push_back(std::move(squared));
                continue;
            }
            Ciphertext score = combine_structural(ic, iw, cw, encoder, evaluator, scale, stats);
            if (scheme == "ours_packaged_structural_ctct") {
                score = package_mean(score, package_size, encoder, evaluator, galois_keys, scale, stats);
            }
            outputs.push_back(std::move(score));
        } else {
            throw std::runtime_error("Unknown scheme: " + scheme);
        }
    }
    return outputs;
}

static std::uint64_t sum_ciphertext_bytes(const std::vector<Ciphertext> &cts)
{
    std::uint64_t total = 0;
    for (const auto &ct : cts) {
        total += static_cast<std::uint64_t>(ciphertext_bytes(ct));
    }
    return total;
}

static std::uint64_t sum_nested_ciphertext_bytes(const std::vector<std::vector<Ciphertext>> &cts)
{
    std::uint64_t total = 0;
    for (const auto &group : cts) {
        total += sum_ciphertext_bytes(group);
    }
    return total;
}

static std::uint64_t count_nested_ciphertexts(const std::vector<std::vector<Ciphertext>> &cts)
{
    std::uint64_t total = 0;
    for (const auto &group : cts) {
        total += static_cast<std::uint64_t>(group.size());
    }
    return total;
}

static double input_prepare_encrypt_ms_for_scheme(const std::string &scheme, const Inputs &in)
{
    if (scheme == "baseline_random_noop") {
        return 0.0;
    }
    if (scheme == "baseline_cosine_ctpt" ||
        scheme == "baseline_linear_student_ctpt" ||
        scheme == "ours_student_row_ctpt" ||
        scheme == "baseline_uncertainty_poly4_ctpt" ||
        scheme == "baseline_coreset_all_distances_ctpt" ||
        scheme == "baseline_badge_components_poly4_ctpt" ||
        scheme == "baseline_kmeans_all_distances_ctpt" ||
        scheme == "baseline_typiclust_sqrt_poly4_ctpt" ||
        scheme == "ours_krr_row_exp_poly4_ctpt") {
        return in.img_prepare_encrypt_ms;
    }
    if (scheme == "baseline_coreset_distance_ctpt") {
        return in.cand_prepare_encrypt_ms;
    }
    if (scheme == "ours_dcc_row_simd_ctpt") {
        return in.dcc_row_prepare_encrypt_ms;
    }
    if (scheme == "ours_dcc_pkg_simd_ctpt") {
        return in.dcc_package_prepare_encrypt_ms;
    }
    if (scheme == "ours_student_pkg_ctpt") {
        return in.student_package_prepare_encrypt_ms;
    }
    if (scheme == "ours_krr_pkg_exp_poly4_ctpt") {
        return in.package_feature_prepare_encrypt_ms;
    }
    if (scheme == "ours_structural_ctpt" || scheme == "ours_packaged_structural_ctpt") {
        return in.img_prepare_encrypt_ms + in.cand_prepare_encrypt_ms;
    }
    if (scheme == "ours_structural_ctct" ||
        scheme == "ours_packaged_structural_ctct" ||
        scheme == "baseline_poly2_fusion_ctct") {
        return in.img_prepare_encrypt_ms + in.weak_prepare_encrypt_ms + in.cand_prepare_encrypt_ms;
    }
    return 0.0;
}

static std::uint64_t input_ciphertexts_for_scheme(const std::string &scheme, const Inputs &in)
{
    if (scheme == "baseline_random_noop") {
        return 0;
    }
    if (scheme == "baseline_cosine_ctpt" ||
        scheme == "baseline_linear_student_ctpt" ||
        scheme == "ours_student_row_ctpt" ||
        scheme == "baseline_uncertainty_poly4_ctpt" ||
        scheme == "baseline_coreset_all_distances_ctpt" ||
        scheme == "baseline_badge_components_poly4_ctpt" ||
        scheme == "baseline_kmeans_all_distances_ctpt" ||
        scheme == "baseline_typiclust_sqrt_poly4_ctpt" ||
        scheme == "ours_krr_row_exp_poly4_ctpt") {
        return count_nested_ciphertexts(in.img_ct);
    }
    if (scheme == "baseline_coreset_distance_ctpt") {
        return count_nested_ciphertexts(in.cand_ct);
    }
    if (scheme == "ours_dcc_row_simd_ctpt") {
        return count_nested_ciphertexts(in.dcc_row_summary_ct);
    }
    if (scheme == "ours_dcc_pkg_simd_ctpt") {
        return count_nested_ciphertexts(in.dcc_package_summary_ct);
    }
    if (scheme == "ours_student_pkg_ctpt") {
        return count_nested_ciphertexts(in.package_summary_ct);
    }
    if (scheme == "ours_krr_pkg_exp_poly4_ctpt") {
        return count_nested_ciphertexts(in.package_feature_ct) +
               static_cast<std::uint64_t>(in.package_norm_term_ct.size());
    }
    if (scheme == "ours_structural_ctpt" || scheme == "ours_packaged_structural_ctpt") {
        return count_nested_ciphertexts(in.img_ct) + count_nested_ciphertexts(in.cand_ct);
    }
    if (scheme == "ours_structural_ctct" ||
        scheme == "ours_packaged_structural_ctct" ||
        scheme == "baseline_poly2_fusion_ctct") {
        return count_nested_ciphertexts(in.img_ct) +
               count_nested_ciphertexts(in.weak_ct) +
               count_nested_ciphertexts(in.cand_ct);
    }
    return 0;
}

static std::uint64_t input_ciphertext_bytes_for_scheme(const std::string &scheme, const Inputs &in)
{
    if (scheme == "baseline_random_noop") {
        return 0;
    }
    if (scheme == "baseline_cosine_ctpt" ||
        scheme == "baseline_linear_student_ctpt" ||
        scheme == "ours_student_row_ctpt" ||
        scheme == "baseline_uncertainty_poly4_ctpt" ||
        scheme == "baseline_coreset_all_distances_ctpt" ||
        scheme == "baseline_badge_components_poly4_ctpt" ||
        scheme == "baseline_kmeans_all_distances_ctpt" ||
        scheme == "baseline_typiclust_sqrt_poly4_ctpt" ||
        scheme == "ours_krr_row_exp_poly4_ctpt") {
        return sum_nested_ciphertext_bytes(in.img_ct);
    }
    if (scheme == "baseline_coreset_distance_ctpt") {
        return sum_nested_ciphertext_bytes(in.cand_ct);
    }
    if (scheme == "ours_dcc_row_simd_ctpt") {
        return sum_nested_ciphertext_bytes(in.dcc_row_summary_ct);
    }
    if (scheme == "ours_dcc_pkg_simd_ctpt") {
        return sum_nested_ciphertext_bytes(in.dcc_package_summary_ct);
    }
    if (scheme == "ours_student_pkg_ctpt") {
        return sum_nested_ciphertext_bytes(in.package_summary_ct);
    }
    if (scheme == "ours_krr_pkg_exp_poly4_ctpt") {
        return sum_nested_ciphertext_bytes(in.package_feature_ct) +
               sum_ciphertext_bytes(in.package_norm_term_ct);
    }
    if (scheme == "ours_structural_ctpt" || scheme == "ours_packaged_structural_ctpt") {
        return sum_nested_ciphertext_bytes(in.img_ct) + sum_nested_ciphertext_bytes(in.cand_ct);
    }
    if (scheme == "ours_structural_ctct" ||
        scheme == "ours_packaged_structural_ctct" ||
        scheme == "baseline_poly2_fusion_ctct") {
        return sum_nested_ciphertext_bytes(in.img_ct) +
               sum_nested_ciphertext_bytes(in.weak_ct) +
               sum_nested_ciphertext_bytes(in.cand_ct);
    }
    return 0;
}

static std::uint64_t decrypt_decode_outputs(
    const std::vector<Ciphertext> &cts,
    Decryptor &decryptor,
    CKKSEncoder &encoder)
{
    std::uint64_t decoded_values = 0;
    Plaintext plain;
    std::vector<double> decoded;
    for (const auto &ct : cts) {
        decryptor.decrypt(ct, plain);
        encoder.decode(plain, decoded);
        decoded_values += static_cast<std::uint64_t>(decoded.size());
    }
    return decoded_values;
}

static int reference_count_for_scheme(const std::string &scheme, const Config &cfg)
{
    if (scheme == "ours_krr_row_exp_poly4_ctpt" ||
        scheme == "ours_krr_pkg_exp_poly4_ctpt") {
        return cfg.krr_landmarks;
    }
    if (scheme == "baseline_coreset_all_distances_ctpt" ||
        scheme == "baseline_badge_components_poly4_ctpt") {
        return cfg.coreset_references;
    }
    if (scheme == "baseline_kmeans_all_distances_ctpt") {
        return cfg.kmeans_centers;
    }
    if (scheme == "baseline_typiclust_sqrt_poly4_ctpt") {
        return cfg.typiclust_neighbors;
    }
    return 0;
}

static int polynomial_degree_for_scheme(const std::string &scheme)
{
    if (scheme == "baseline_uncertainty_poly4_ctpt" ||
        scheme == "baseline_badge_components_poly4_ctpt" ||
        scheme == "baseline_typiclust_sqrt_poly4_ctpt" ||
        scheme == "ours_krr_row_exp_poly4_ctpt" ||
        scheme == "ours_krr_pkg_exp_poly4_ctpt") {
        return 4;
    }
    return 0;
}

static int ciphertext_nonlinear_depth_for_scheme(const std::string &scheme)
{
    return polynomial_degree_for_scheme(scheme) == 4 ? 2 : 0;
}

int main(int argc, char **argv)
{
    try {
        Config cfg = parse_args(argc, argv);
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(cfg.poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(cfg.poly_modulus_degree, cfg.coeff_modulus_bits));
        auto context = SEALContext(parms);
        if (!context.parameters_set()) {
            throw std::runtime_error("Invalid SEAL parameters.");
        }

        const double scale = std::pow(2.0, cfg.scale_bits);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        GaloisKeys galois_keys;
        keygen.create_galois_keys(galois_keys);

        Encryptor encryptor(context, public_key);
        Evaluator evaluator(context);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        const std::size_t slots = encoder.slot_count();

        std::ofstream out(cfg.out_csv);
        if (!out) {
            throw std::runtime_error("Cannot open output CSV: " + cfg.out_csv);
        }
        out << "scheme,poly_modulus_degree,coeff_modulus_bits,scale_bits,slot_count,logical_rows,chunks,"
            << "scored_objects,raw_rows_per_scored_object,feature_dim,student_summary_dim,package_size,repeat,"
            << "input_prepare_encrypt_ms,encrypted_compute_ms,decrypt_decode_ms,total_full_flow_ms,"
            << "threshold_parties,rows_per_second,scored_objects_per_second,output_ciphertexts,"
            << "decoded_values,input_ciphertexts,input_ciphertext_bytes,output_ciphertext_bytes,"
            << "total_communication_bytes,ct_ct_mults,ct_pt_mults,rotations,additions,"
            << "relinearizations,rescales,reference_count,poly_degree,ctct_nonlinear_depth\n";

        std::mt19937_64 rng(cfg.seed);
        for (int logical_rows : cfg.rows) {
            if (logical_rows <= 0) {
                continue;
            }
            const std::size_t chunks = static_cast<std::size_t>((logical_rows + static_cast<int>(slots) - 1) / static_cast<int>(slots));
            for (int dim : cfg.dims) {
                if (dim <= 0) {
                    continue;
                }
                std::cout << "[prepare] rows=" << logical_rows << " dim=" << dim
                          << " chunks=" << chunks << " slots=" << slots
                          << " package_size=" << cfg.package_size
                          << " student_summary_dim=" << cfg.student_summary_dim << std::endl;
                Inputs inputs = prepare_inputs(
                    logical_rows, dim, cfg.student_summary_dim, cfg.package_size,
                    slots, encoder, encryptor, scale, rng);

                for (const auto &scheme : cfg.schemes) {
                    std::cout << "[scheme] " << scheme << std::endl;
                    for (int warmup = 0; warmup < cfg.warmups; ++warmup) {
                        Stats warmup_stats;
                        (void)run_scheme(
                            scheme, inputs, cfg.package_size, cfg.krr_landmarks,
                            cfg.coreset_references, cfg.kmeans_centers,
                            cfg.typiclust_neighbors, encoder, evaluator,
                            relin_keys, galois_keys, scale, warmup_stats);
                    }
                    for (int rep = 0; rep < cfg.repeats; ++rep) {
                        Stats stats;
                        auto start = Clock::now();
                        std::vector<Ciphertext> outputs = run_scheme(
                            scheme, inputs, cfg.package_size, cfg.krr_landmarks,
                            cfg.coreset_references, cfg.kmeans_centers,
                            cfg.typiclust_neighbors, encoder, evaluator,
                            relin_keys, galois_keys, scale, stats);
                        auto end = Clock::now();
                        const double elapsed_ms =
                            std::chrono::duration<double, std::milli>(end - start).count();
                        const double prepare_encrypt_ms = input_prepare_encrypt_ms_for_scheme(scheme, inputs);
                        double decrypt_decode_ms = 0.0;
                        std::uint64_t decoded_values = 0;
                        if ((cfg.validate || cfg.measure_decrypt_all) && !outputs.empty()) {
                            auto decrypt_start = Clock::now();
                            if (cfg.measure_decrypt_all) {
                                decoded_values = decrypt_decode_outputs(outputs, decryptor, encoder);
                            } else {
                                Plaintext plain;
                                std::vector<double> decoded;
                                decryptor.decrypt(outputs.front(), plain);
                                encoder.decode(plain, decoded);
                                decoded_values = static_cast<std::uint64_t>(decoded.size());
                                if (decoded.empty() || !std::isfinite(decoded.front())) {
                                    throw std::runtime_error("Validation failed: non-finite decoded value.");
                                }
                            }
                            auto decrypt_end = Clock::now();
                            decrypt_decode_ms =
                                std::chrono::duration<double, std::milli>(decrypt_end - decrypt_start).count();
                        }
                        const double total_full_flow_ms =
                            prepare_encrypt_ms + elapsed_ms + decrypt_decode_ms;
                        const double rows_per_second =
                            static_cast<double>(logical_rows) / std::max(1e-9, elapsed_ms / 1000.0);
                        const bool package_scoring =
                            (scheme == "ours_student_pkg_ctpt" ||
                             scheme == "ours_dcc_pkg_simd_ctpt" ||
                             scheme == "ours_krr_pkg_exp_poly4_ctpt");
                        const int scored_objects = package_scoring ? inputs.scored_packages : logical_rows;
                        const double scored_objects_per_second =
                            static_cast<double>(scored_objects) / std::max(1e-9, elapsed_ms / 1000.0);
                        const std::uint64_t bytes = sum_ciphertext_bytes(outputs);
                        const std::uint64_t input_cts = input_ciphertexts_for_scheme(scheme, inputs);
                        const std::uint64_t input_bytes = input_ciphertext_bytes_for_scheme(scheme, inputs);
                        const std::uint64_t total_communication_bytes = input_bytes + bytes;

                        out << scheme << ','
                            << cfg.poly_modulus_degree << ','
                            << join_ints(cfg.coeff_modulus_bits) << ','
                            << cfg.scale_bits << ','
                            << slots << ','
                            << logical_rows << ','
                            << chunks << ','
                            << scored_objects << ','
                            << (package_scoring ? cfg.package_size : 1) << ','
                            << dim << ','
                            << cfg.student_summary_dim << ','
                            << cfg.package_size << ','
                            << rep << ','
                            << std::fixed << std::setprecision(6) << prepare_encrypt_ms << ','
                            << std::fixed << std::setprecision(6) << elapsed_ms << ','
                            << std::fixed << std::setprecision(6) << decrypt_decode_ms << ','
                            << std::fixed << std::setprecision(6) << total_full_flow_ms << ','
                            << cfg.threshold_parties << ','
                            << std::fixed << std::setprecision(6) << rows_per_second << ','
                            << std::fixed << std::setprecision(6) << scored_objects_per_second << ','
                            << outputs.size() << ','
                            << decoded_values << ','
                            << input_cts << ','
                            << input_bytes << ','
                            << bytes << ','
                            << total_communication_bytes << ','
                            << stats.ct_ct_mults << ','
                            << stats.ct_pt_mults << ','
                            << stats.rotations << ','
                            << stats.additions << ','
                            << stats.relinearizations << ','
                            << stats.rescales << ','
                            << reference_count_for_scheme(scheme, cfg) << ','
                            << polynomial_degree_for_scheme(scheme) << ','
                            << ciphertext_nonlinear_depth_for_scheme(scheme) << '\n';
                        out.flush();
                    }
                }
            }
        }
        std::cout << "[done] " << cfg.out_csv << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }
}
