#ifndef BP_H
#define BP_H

#include <utility>
#include <vector>
#include <memory>
#include <iterator>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <stdexcept> // required for std::runtime_error
#include <set>

#include "math.h"
#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"
#include "rng.hpp"

namespace ldpc {
    namespace bp {

        enum BpMethod {
            PRODUCT_SUM = 0,
            MINIMUM_SUM = 1
        };

        enum BpSchedule {
            SERIAL = 0,
            PARALLEL = 1,
            SERIAL_RELATIVE = 2,
            CLUSTER = 3
        };

        enum BpInputType {
            SYNDROME = 0,
            RECEIVED_VECTOR = 1,
            AUTO = 2
        };

        const std::vector<int> NULL_INT_VECTOR = {};

        class BpEntry : public ldpc::sparse_matrix_base::EntryBase<BpEntry> {
        public:
            double bit_to_check_msg = 0.0;
            double check_to_bit_msg = 0.0;

            ~BpEntry() = default;
        };
        using BpSparse = ldpc::gf2sparse::GF2Sparse<BpEntry>;

        class BpDecoder {
            // TODO properties should be private and only accessible via getters and setters
        public:
            BpSparse &pcm;
            std::vector<double> channel_probabilities;
            int check_count;
            int bit_count;
            int maximum_iterations;
            BpMethod bp_method;
            BpSchedule schedule;
            BpInputType bp_input_type;
            double ms_scaling_factor;
            std::vector<uint8_t> decoding;
            std::vector<uint8_t> candidate_syndrome;

            std::vector<double> log_prob_ratios;
            std::vector<double> initial_log_prob_ratios;
            std::vector<double> soft_syndrome;
            std::vector<int> serial_schedule_order;
            int iterations;
            int omp_thread_count;
            bool converge;
            int random_schedule_seed;
            bool random_schedule_at_every_iteration;
            ldpc::rng::RandomListShuffle<int> rng_list_shuffle;

            BpDecoder(
                    BpSparse &parity_check_matrix,
                    std::vector<double> channel_probabilities,
                    int maximum_iterations = 0,
                    BpMethod bp_method = PRODUCT_SUM,
                    BpSchedule schedule = PARALLEL,
                    double min_sum_scaling_factor = 0.625,
                    int omp_threads = 1,
                    const std::vector<int> &serial_schedule = NULL_INT_VECTOR,
                    int random_schedule_seed = -1, // TODO what should be default here? 0 is set but -1 is checked in decode method?
                    bool random_schedule_at_every_iteration = true,
                    BpInputType bp_input_type = AUTO) :
                    pcm(parity_check_matrix), channel_probabilities(std::move(channel_probabilities)),
                    check_count(pcm.m), bit_count(pcm.n), maximum_iterations(maximum_iterations), bp_method(bp_method),
                    schedule(schedule), ms_scaling_factor(min_sum_scaling_factor),
                    iterations(0) //the parity check matrix is passed in by reference
            {

                this->initial_log_prob_ratios.resize(bit_count);
                this->log_prob_ratios.resize(bit_count);
                this->candidate_syndrome.resize(check_count);
                this->decoding.resize(bit_count);
                this->converge = 0;
                this->omp_thread_count = omp_threads;
                this->random_schedule_seed = random_schedule_seed;
                this->random_schedule_at_every_iteration = random_schedule_at_every_iteration;
                this->bp_input_type = bp_input_type;


                if (this->channel_probabilities.size() != this->bit_count) {
                    throw std::runtime_error(
                            "Channel probabilities vector must have length equal to the number of bits");
                }
                if (serial_schedule != NULL_INT_VECTOR) {
                    this->serial_schedule_order = serial_schedule;
                    this->random_schedule_seed = -1;
                } else {
                    this->serial_schedule_order.resize(bit_count);
                    for (int i = 0; i < bit_count; i++) {
                        this->serial_schedule_order[i] = i;
                    }
                    this->rng_list_shuffle.seed(this->random_schedule_seed);
                }

                //Initialise OMP thread pool
                // this->omp_thread_count = omp_threads;
                // this->set_omp_thread_count(this->omp_thread_count);
            }

            ~BpDecoder() = default;

            void set_omp_thread_count(int count) {
                this->omp_thread_count = count;
                // omp_set_num_threads(this->omp_thread_count);
                // NotImplemented
            }

            /**
             * Resets iteration count, convergence flag, decoding/messages and LLRs to zero
             * without reallocating, so the decoder can be reused across successive calls
             * to `initialise_log_domain_bp(llr_vector)` + `bp_decode_cluster(...)`.
             */
            void reset() {
                this->iterations = 0;
                this->converge = false;

                std::fill(this->decoding.begin(), this->decoding.end(), 0);
                std::fill(this->candidate_syndrome.begin(), this->candidate_syndrome.end(), 0);
                std::fill(this->log_prob_ratios.begin(), this->log_prob_ratios.end(), 0.0);
                std::fill(this->initial_log_prob_ratios.begin(), this->initial_log_prob_ratios.end(), 0.0);

                for (int i = 0; i < this->bit_count; i++) {
                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = 0.0;
                        e.check_to_bit_msg = 0.0;
                    }
                }
            }

            /**
             * Seeds bit-to-check messages and log-probability ratios directly from an
             * externally supplied per-bit channel LLR vector, bypassing `channel_probabilities`.
             * This is an additive overload alongside the original no-arg
             * `initialise_log_domain_bp()`; it does not change that method's behaviour.
             */
            void initialise_log_domain_bp(const std::vector<double> &llr_vector_channel) {
                for (int i = 0; i < this->bit_count; i++) {
                    this->initial_log_prob_ratios[i] = llr_vector_channel[i];
                    this->log_prob_ratios[i] = llr_vector_channel[i];

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = llr_vector_channel[i];
                        e.check_to_bit_msg = 0.0;
                    }
                }
            }

            void initialise_log_domain_bp() {
                // initialise BP
                for (int i = 0; i < this->bit_count; i++) {
                    this->initial_log_prob_ratios[i] = std::log(
                            (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = this->initial_log_prob_ratios[i];
                    }
                }
            }

            std::vector<uint8_t> decode(std::vector<uint8_t> &input_vector) {


                if ((this->bp_input_type == AUTO && input_vector.size() == this->bit_count) ||
                    this->bp_input_type == RECEIVED_VECTOR) {
                    auto syndrome = pcm.mulvec(input_vector);
                    std::vector<uint8_t> rv_decoding;
                    if (schedule == PARALLEL) {
                        rv_decoding = bp_decode_parallel(syndrome);
                    } else if (schedule == SERIAL || schedule == SERIAL_RELATIVE) {
                        rv_decoding = bp_decode_serial(syndrome);
                    } else {
                        throw std::runtime_error("Invalid BP schedule");
                    }

                    for (int i = 0; i < this->bit_count; i++) {
                        this->decoding[i] = rv_decoding[i] ^ input_vector[i];
                    }

                    return this->decoding;

                }


                if (schedule == PARALLEL) {
                    return bp_decode_parallel(input_vector);
                }
                if (schedule == SERIAL || schedule == SERIAL_RELATIVE) {
                    return bp_decode_serial(input_vector);
                } else { throw std::runtime_error("Invalid BP schedule"); }

            }

            /**
             * Runs a single product-sum BP update restricted to the given subset of check
             * nodes (`cluster_checks`), operating on the current `log_prob_ratios`/message
             * state (seeded via `initialise_log_domain_bp(llr_vector)` and advanced via
             * repeated calls to this method, typically interleaved across several clusters
             * per "iteration"). This is a new, additive entry point alongside the existing
             * `decode()`/`bp_decode_parallel()`/`bp_decode_serial()` schedules; it does not
             * replace or alter them.
             */
            void bp_decode_cluster(const std::vector<int> &cluster_checks) {
                if (cluster_checks.empty()) {
                    return;
                }

                std::vector<uint8_t> check_mask(check_count, 0);
                for (int check_index: cluster_checks) {
                    if (check_index < 0 || check_index >= check_count) {
                        throw std::runtime_error("Cluster contains invalid check index");
                    }
                    check_mask[check_index] = 1;
                }

                const double EPS_TANH = 1e-12;

                for (int col = 0; col < this->bit_count; ++col) {
                    for (auto &edge: pcm.iterate_column(col)) {
                        if (check_mask[edge.row_index]) {
                            edge.bit_to_check_msg = this->log_prob_ratios[col] - edge.check_to_bit_msg;
                        }
                    }
                }

                if (bp_method == PRODUCT_SUM) {
                    for (int check_index: cluster_checks) {
                        double Am = 0.0;
                        for (auto &edge: pcm.iterate_row(check_index)) {
                            double t = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t) < EPS_TANH) {
                                t = (t >= 0) ? EPS_TANH : -EPS_TANH;
                            }
                            Am += std::log(std::abs(t));
                        }

                        int sm = 1;
                        for (auto &edge: pcm.iterate_row(check_index)) {
                            if (edge.bit_to_check_msg < 0.0) {
                                sm = -sm;
                            }
                        }

                        for (auto &edge: pcm.iterate_row(check_index)) {
                            double oldR = edge.check_to_bit_msg;

                            double t_self = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t_self) < EPS_TANH) {
                                t_self = (t_self >= 0.0 ? EPS_TANH : -EPS_TANH);
                            }

                            double log_abs_t_self = std::log(std::abs(t_self));
                            double temp = Am - log_abs_t_self;

                            int sign_Lmj = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
                            int sign_factor = sm * sign_Lmj;
                            double prod_others = sign_factor * std::exp(temp);

                            if (prod_others > 1.0 - 1e-15) {
                                prod_others = 1.0 - 1e-15;
                            }
                            if (prod_others < -1.0 + 1e-15) {
                                prod_others = -1.0 + 1e-15;
                            }

                            double newR = std::log((1.0 + prod_others) / (1.0 - prod_others));

                            if (!std::isfinite(newR)) {
                                if (std::isnan(newR)) {
                                    newR = 0.0;
                                } else if (newR > 1e300) {
                                    newR = 1e300;
                                } else if (newR < -1e300) {
                                    newR = -1e300;
                                }
                            }

                            edge.check_to_bit_msg = newR;
                            this->log_prob_ratios[edge.col_index] += (newR - oldR);
                        }
                    }
                } else {
                    throw std::runtime_error("Cluster decoding with Minimum-Sum method is not yet implemented");
                }
            }

            /**
             * Per-check-node residual: the maximum absolute change a check node's outgoing
             * messages would undergo if it were updated now, given the current
             * bit-to-check messages. Used as a state/reward signal for scheduling which
             * cluster to update next; does not mutate decoder state.
             */
            std::vector<double> get_residuals() {
                std::vector<double> residuals(this->check_count, 0.0);

                if (this->bp_method == PRODUCT_SUM) {
                    for (int row = 0; row < this->check_count; ++row) {
                        double max_residual = 0.0;
                        const double EPS_TANH = 1e-12;
                        double Am = 0.0;
                        int sm = 1;
                        for (auto &edge: pcm.iterate_row(row)) {
                            double t = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t) < EPS_TANH) {
                                t = (t >= 0) ? EPS_TANH : -EPS_TANH;
                            }
                            Am += std::log(std::abs(t));
                            if (edge.bit_to_check_msg < 0.0) sm = -sm;
                        }

                        for (auto &edge: pcm.iterate_row(row)) {
                            double old_msg = edge.check_to_bit_msg;

                            double t_self = std::tanh(edge.bit_to_check_msg / 2.0);
                            if (std::abs(t_self) < EPS_TANH) {
                                t_self = (t_self >= 0.0 ? EPS_TANH : -EPS_TANH);
                            }
                            double log_abs_t_self = std::log(std::abs(t_self));
                            double temp = Am - log_abs_t_self;

                            int sign_Lmj = (edge.bit_to_check_msg < 0.0) ? -1 : 1;
                            int sign_factor = sm * sign_Lmj;
                            double prod_others = sign_factor * std::exp(temp);

                            if (prod_others > 1.0 - 1e-15) {
                                prod_others = 1.0 - 1e-15;
                            }
                            if (prod_others < -1.0 + 1e-15) {
                                prod_others = -1.0 + 1e-15;
                            }

                            double new_msg = std::log((1.0 + prod_others) / (1.0 - prod_others));
                            double residual = std::abs(new_msg - old_msg);
                            if (residual > max_residual) {
                                max_residual = residual;
                            }
                        }
                        residuals[row] = max_residual;
                    }
                } else if (this->bp_method == MINIMUM_SUM) {
                    throw std::runtime_error("Minsum is not implemented yet");
                }

                return residuals;
            }

            /**
             * Gaussian-approximation J-function (EXIT-chart mutual information as a
             * function of the LLR channel standard deviation), per ten Brink's
             * approximation. Used by `m2i2_scheduler`.
             */
            double J_func(const double sigma) {
                double mi = 0.0;

                if (sigma >= 10) {
                    mi = 1.0;
                }
                else if (sigma > 1.6363) {
                    mi = 1.0 - std::exp(0.001815 * sigma * sigma * sigma - 0.142675 * sigma * sigma - 0.082205 * sigma + 0.054960);
                }
                else {
                    mi = -0.0421061 * sigma * sigma * sigma + 0.209252 * sigma * sigma - 0.00640081 * sigma;
                }

                return mi;
            }

            /** Inverse of `J_func`: recovers the LLR standard deviation from a mutual information value. */
            double J_inv_func(const double mi) {
                double sigma = 0.0;

                if (mi <= 0.3646) {
                    sigma = 1.09542 * mi * mi + 0.214217 * mi + 2.33727 * std::sqrt(mi);
                }

                else {
                    sigma = -0.706692 * std::log(0.386013 * (1.0 - mi)) - 1.75017 * mi;
                }

                return sigma;
            }

            /**
             * Computes a check-node update schedule for a base (protograph) matrix `P`
             * using EXIT-chart mutual-information tracking (the "M2I2" heuristic):
             * greedily orders check-node updates by expected mutual-information gain
             * under a Gaussian-approximation BP model, given the code rate and channel
             * Eb/N0. Independent of the main decode()/bp_decode_cluster() message-passing
             * state; operates purely on the supplied base matrix and EXIT-chart model.
             */
            std::vector<int> m2i2_scheduler(const std::vector<std::vector<int>> &P, double code_rate, double EbN0, int max_iterations) {
                std::vector<int> schedule;

                int Mp = P.size();
                if (Mp == 0) {
                    throw std::runtime_error("Base matrix P is empty");
                }
                int Np = P[0].size();

                std::vector<std::vector<int>> u(Mp, std::vector<int>(Np, 0));
                std::vector<std::vector<double>> I_EC(Mp, std::vector<double>(Np, 0.0));
                std::vector<std::vector<double>> I_EV(Mp, std::vector<double>(Np, 0.0));
                std::vector<double> I_ch(Np, 0.0);
                std::vector<std::vector<double>> Ip_EC(Mp, std::vector<double>(Np, 0.0));
                std::vector<double> R_cluster(Mp, 0.0);
                std::vector<double> I_CMI(Np, 0.0);

                double sigma_ch = std::sqrt(8.0 * code_rate * EbN0);
                for (int j = 0; j < Np; ++j) {
                    I_ch[j] = J_func(sigma_ch);
                }

                for (int i = 0; i < Mp; ++i) {
                    for (int j = 0; j < Np; ++j) {
                        if (P[i][j] != -1) {
                            I_EV[i][j] = I_ch[j];
                        }
                    }
                }
                while (schedule.size() < max_iterations) {
                    for (int i = 0; i < Mp; ++i) {
                        for (int j = 0; j < Np; ++j) {
                            if (P[i][j] != -1) {
                                double sum_sq = 0.0;
                                for (int b = 0; b < Np; ++b) {
                                    if (b != j && P[i][b] != -1) {
                                        double ji = J_inv_func(1.0 - I_EV[i][b]);
                                        sum_sq += ji * ji;
                                    }
                                }
                                Ip_EC[i][j] = 1.0 - J_func(std::sqrt(sum_sq));
                            }
                        }
                    }

                    for (int i = 0; i < Mp; ++i) {
                        R_cluster[i] = 0.0;
                        for (int j = 0; j < Np; ++j) {
                            if (P[i][j] != -1) {
                                R_cluster[i] += Ip_EC[i][j] - I_EC[i][j];
                            }
                        }
                    }

                    int i_star = 0;
                    double max_increase = R_cluster[0];
                    for (int i = 1; i < Mp; ++i) {
                        if (R_cluster[i] > max_increase) {
                            max_increase = R_cluster[i];
                            i_star = i;
                        }
                    }

                    schedule.push_back(i_star);

                    for (int j = 0; j < Np; ++j) {
                        if (P[i_star][j] != -1) {
                            I_EC[i_star][j] = Ip_EC[i_star][j];
                            u[i_star][j] += 1;
                        }
                    }

                    for (int j = 0; j < Np; ++j) {
                        if (P[i_star][j] != -1) {
                            for (int a = 0; a < Mp; ++a) {
                                if (P[a][j] != -1) {
                                    double sum_sq = 0.0;
                                    for (int c = 0; c < Mp; ++c) {
                                        if (c != a && P[c][j] != -1) {
                                            double ji = J_inv_func(I_EC[c][j]);
                                            sum_sq += ji * ji;
                                        }
                                    }

                                    double ji_ch = J_inv_func(I_ch[j]);
                                    sum_sq += ji_ch * ji_ch;
                                    I_EV[a][j] = J_func(std::sqrt(sum_sq));
                                }
                            }
                        }
                    }

                    bool converged = true;
                    for (int j = 0; j < Np; ++j) {
                        double sum_sq = 0.0;
                        for (int a = 0; a < Mp; ++a) {
                            if (P[a][j] != -1) {
                                double ji = J_inv_func(I_EC[a][j]);
                                sum_sq += ji * ji;
                            }
                        }

                        double ji_ch = J_inv_func(I_ch[j]);
                        sum_sq += ji_ch * ji_ch;
                        I_CMI[j] = J_func(std::sqrt(sum_sq));

                        if (I_CMI[j] < 1.0) {
                            converged = false;
                        }
                    }

                    if (converged) {
                        break;
                    }
                }

                return schedule;
            }

            std::vector<uint8_t> &bp_decode_parallel(std::vector<uint8_t> &syndrome) {

                this->converge = 0;

                this->initialise_log_domain_bp();

                //main interation loop
                for (int it = 1; it <= this->maximum_iterations; it++) {

                    if (this->bp_method == PRODUCT_SUM) {
                        for (int i = 0; i < this->check_count; i++) {
                            this->candidate_syndrome[i] = 0;

                            double temp = 1.0;
                            for (auto &e: this->pcm.iterate_row(i)) {
                                e.check_to_bit_msg = temp;
                                temp *= std::tanh(e.bit_to_check_msg / 2);
                            }

                            temp = 1;
                            for (auto &e: this->pcm.reverse_iterate_row(i)) {
                                e.check_to_bit_msg *= temp;
                                int message_sign = syndrome[i] != 0u ? -1.0 : 1.0;
                                e.check_to_bit_msg =
                                        message_sign * std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));
                                temp *= std::tanh(e.bit_to_check_msg / 2);
                            }
                        }
                    } else if (this->bp_method == MINIMUM_SUM) {
                        //check to bit updates
                        for (int i = 0; i < check_count; i++) {

                            this->candidate_syndrome[i] = 0;
                            int total_sgn = 0;
                            int sgn = 0;
                            total_sgn = syndrome[i];
                            double temp = std::numeric_limits<double>::max();

                            for (auto &e: this->pcm.iterate_row(i)) {
                                if (e.bit_to_check_msg <= 0) {
                                    total_sgn += 1;
                                }
                                e.check_to_bit_msg = temp;
                                double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                if (abs_bit_to_check_msg < temp) {
                                    temp = abs_bit_to_check_msg;
                                }
                            }

                            temp = std::numeric_limits<double>::max();
                            for (auto &e: this->pcm.reverse_iterate_row(i)) {
                                sgn = total_sgn;
                                if (e.bit_to_check_msg <= 0) {
                                    sgn += 1;
                                }
                                if (temp < e.check_to_bit_msg) {
                                    e.check_to_bit_msg = temp;
                                }

                                int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                                e.check_to_bit_msg *= message_sign * ms_scaling_factor;

                                double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                if (abs_bit_to_check_msg < temp) {
                                    temp = abs_bit_to_check_msg;
                                }

                            }

                        }
                    }


                    //compute log probability ratios
                    for (int i = 0; i < this->bit_count; i++) {
                        double temp = initial_log_prob_ratios[i];
                        for (auto &e: this->pcm.iterate_column(i)) {
                            e.bit_to_check_msg = temp;
                            temp += e.check_to_bit_msg;
                            // if(isnan(temp)) temp = e.bit_to_check_msg;


                        }

                        //make hard decision on basis of log probability ratio for bit i
                        this->log_prob_ratios[i] = temp;
                        // if(isnan(log_prob_ratios[i])) log_prob_ratios[i] = initial_log_prob_ratios[i];
                        if (temp <= 0) {
                            this->decoding[i] = 1;
                            for (auto &e: this->pcm.iterate_column(i)) {
                                this->candidate_syndrome[e.row_index] ^= 1;
                            }
                        } else {
                            this->decoding[i] = 0;
                        }
                    }

                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        this->converge = true;
                    }

                    this->iterations = it;

                    if (this->converge) {
                        return this->decoding;
                    }


                    //compute bit to check update
                    for (int i = 0; i < bit_count; i++) {
                        double temp = 0;
                        for (auto &e: this->pcm.reverse_iterate_column(i)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }

                }


                return this->decoding;

            }

            std::vector<uint8_t> &bp_decode_single_scan(std::vector<uint8_t> &syndrome) {

                converge = 0;
                int CONVERGED = 0;

                std::vector<double> log_prob_ratios_old;
                log_prob_ratios_old.resize(bit_count);

                for (int i = 0; i < bit_count; i++) {
                    this->initial_log_prob_ratios[i] = std::log(
                            (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);
                    this->log_prob_ratios[i] = this->initial_log_prob_ratios[i];

                }

                // initialise_log_domain_bp();

                //main interation loop
                for (int it = 1; it <= maximum_iterations; it++) {

                    if (CONVERGED != 0) {
                        continue;
                    }

                    // std::fill(candidate_syndrome.begin(), candidate_syndrome.end(), 0);

                    log_prob_ratios_old = this->log_prob_ratios;

                    if (it != 1) {
                        this->log_prob_ratios = this->initial_log_prob_ratios;
                    }

                    //check to bit updates
                    for (int i = 0; i < check_count; i++) {

                        this->candidate_syndrome[i] = 0;

                        int total_sgn = 0;
                        int sgn = 0;
                        total_sgn = syndrome[i];
                        double temp = std::numeric_limits<double>::max();

                        double bit_to_check_msg = NAN;

                        for (auto &e: pcm.iterate_row(i)) {
                            if (it == 1) {
                                e.check_to_bit_msg = 0;
                            }
                            bit_to_check_msg = log_prob_ratios_old[e.col_index] - e.check_to_bit_msg;
                            if (bit_to_check_msg <= 0) {
                                total_sgn += 1;
                            }
                            e.bit_to_check_msg = temp;
                            double abs_bit_to_check_msg = std::abs(bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }

                        temp = std::numeric_limits<double>::max();
                        for (auto &e: pcm.reverse_iterate_row(i)) {
                            sgn = total_sgn;
                            if (it == 1) {
                                e.check_to_bit_msg = 0;
                            }
                            bit_to_check_msg = log_prob_ratios_old[e.col_index] - e.check_to_bit_msg;
                            if (bit_to_check_msg <= 0) {
                                sgn += 1;
                            }
                            if (temp < e.bit_to_check_msg) {
                                e.bit_to_check_msg = temp;
                            }

                            int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                            e.check_to_bit_msg = message_sign * ms_scaling_factor * e.bit_to_check_msg;
                            this->log_prob_ratios[e.col_index] += e.check_to_bit_msg;


                            double abs_bit_to_check_msg = std::abs(bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }

                        }


                    }



                    //compute hard decisions and calculate syndrome
                    for (int i = 0; i < bit_count; i++) {
                        if (this->log_prob_ratios[i] <= 0) {
                            this->decoding[i] = 1;
                            for (auto &e: pcm.iterate_column(i)) {
                                this->candidate_syndrome[e.row_index] ^= 1;
                            }
                        } else {
                            this->decoding[i] = 0;
                        }
                    }

                    int loop_break = 0;
                    CONVERGED = 0;

                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        CONVERGED = 1;
                    }

                    iterations = it;

                    if (CONVERGED != 0) {
                        converge = (CONVERGED != 0);
                        return decoding;
                    }

                }


                converge = (CONVERGED != 0);
                return decoding;

            }

            std::vector<uint8_t> &bp_decode_serial(std::vector<uint8_t> &syndrome) {
                int check_index = 0;
                this->converge = false;
                // initialise BP
                this->initialise_log_domain_bp();

                for (int it = 1; it <= maximum_iterations; it++) {
                    if (this->random_schedule_seed > -1) {
                        this->rng_list_shuffle.shuffle(this->serial_schedule_order);
                    } else if (this->schedule == BpSchedule::SERIAL_RELATIVE) {
                        // resort by LLRs in each iteration to ensure that the most reliable bits are considered first
                        std::sort(this->serial_schedule_order.begin(), this->serial_schedule_order.end(),
                                  [this, it](int bit1, int bit2) {
                                      if (it != 1) {
                                          return this->log_prob_ratios[bit1] > this->log_prob_ratios[bit2];
                                      } else {
                                          return std::log(
                                                  (1 - channel_probabilities[bit1]) / channel_probabilities[bit1]) >
                                                 std::log((1 - channel_probabilities[bit2]) /
                                                          channel_probabilities[bit2]);
                                      }
                                  });
                    }

                    for (int bit_index: this->serial_schedule_order) {
                        double temp = NAN;
                        this->log_prob_ratios[bit_index] = std::log(
                                (1 - channel_probabilities[bit_index]) / channel_probabilities[bit_index]);
                        if (this->bp_method == 0) {
                            for (auto &e: this->pcm.iterate_column(bit_index)) {
                                check_index = e.row_index;
                                e.check_to_bit_msg = 1.0;
                                for (auto &g: this->pcm.iterate_row(check_index)) {
                                    if (&g != &e) {
                                        e.check_to_bit_msg *= tanh(g.bit_to_check_msg / 2);
                                    }
                                }
                                e.check_to_bit_msg = pow(-1, syndrome[check_index]) *
                                                     std::log((1 + e.check_to_bit_msg) / (1 - e.check_to_bit_msg));
                                e.bit_to_check_msg = log_prob_ratios[bit_index];
                                this->log_prob_ratios[bit_index] += e.check_to_bit_msg;
                            }
                        } else if (this->bp_method == 1) {
                            for (auto &e: pcm.iterate_column(bit_index)) {
                                check_index = e.row_index;
                                int sgn = syndrome[check_index];
                                temp = std::numeric_limits<double>::max();
                                for (auto &g: this->pcm.iterate_row(check_index)) {
                                    if (&g != &e) {
                                        double abs_bit_to_check_msg = std::abs(g.bit_to_check_msg);
                                        if (abs_bit_to_check_msg < temp) {
                                            temp = abs_bit_to_check_msg;
                                        }
                                        if (g.bit_to_check_msg <= 0) {
                                            sgn += 1;
                                        }
                                    }
                                }
                                double message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                                e.check_to_bit_msg = ms_scaling_factor * message_sign * temp;
                                e.bit_to_check_msg = log_prob_ratios[bit_index];
                                this->log_prob_ratios[bit_index] += e.check_to_bit_msg;
                            }
                        }
                        if (this->log_prob_ratios[bit_index] <= 0) {
                            this->decoding[bit_index] = 1;
                        } else {
                            this->decoding[bit_index] = 0;
                        }
                        temp = 0;
                        for (auto &e: this->pcm.reverse_iterate_column(bit_index)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }

                    // compute the syndrome for the current candidate decoding solution
                    this->candidate_syndrome = pcm.mulvec(decoding, candidate_syndrome);
                    this->iterations = it;
                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        this->converge = true;
                        return this->decoding;
                    }
                }
                return this->decoding;
            }

            std::vector<uint8_t> &
            soft_info_decode_serial(std::vector<double> &soft_info_syndrome, double cutoff, double sigma) {
                // compute the syndrome log-likelihoods and initialize hard syndrome
                std::vector<uint8_t> syndrome;
                this->soft_syndrome = soft_info_syndrome;
                for (int i = 0; i < this->check_count; i++) {
                    this->soft_syndrome[i] = 2 * this->soft_syndrome[i] / (sigma * sigma);
                    if (this->soft_syndrome[i] <= 0) {
                        syndrome.push_back(1);
                    } else {
                        syndrome.push_back(0);
                    }
                }

                int check_index = 0;
                this->converge = false;
                bool CONVERGED = false;
                bool loop_break = false;
                // initialise BP
                this->initialise_log_domain_bp();
                std::set<int> check_indices_updated;

                for (int it = 1; it <= maximum_iterations; it++) {
                    if (CONVERGED) {
                        continue;
                    }
                    if (this->random_schedule_at_every_iteration && omp_thread_count == 1) {
                        // reorder schedule elements randomly
                        shuffle(serial_schedule_order.begin(), serial_schedule_order.end(),
                                std::default_random_engine(random_schedule_seed));
                    }

                    check_indices_updated.clear();
                    for (auto bit_index: serial_schedule_order) {
                        double temp = NAN;
                        log_prob_ratios[bit_index] = std::log(
                                (1 - channel_probabilities[bit_index]) / channel_probabilities[bit_index]);
                        for (auto &check_nbr: pcm.iterate_column(bit_index)) {
                            // first, we compute the min absolute value of neighbours excluding the current recipient
                            check_index = check_nbr.row_index;
                            int sgn = 0;
                            temp = std::numeric_limits<double>::max();
                            for (auto &g: pcm.iterate_row(check_index)) {
                                if (&g != &check_nbr) {
                                    if (std::abs(g.bit_to_check_msg) < temp) {
                                        temp = std::abs(g.bit_to_check_msg);
                                    }
                                    if (g.bit_to_check_msg <= 0) {
                                        sgn ^= 1;
                                    }
                                }
                            }
                            double min_bit_to_check_msg = temp;
                            double propagated_msg = min_bit_to_check_msg;
                            double soft_syndrome_magnitude = std::abs(this->soft_syndrome[check_index]);

                            // if the soft syndrome magnitude is below cutoff, we apply the virtual update rules
                            if (soft_syndrome_magnitude < cutoff) {
                                if (soft_syndrome_magnitude < std::abs(min_bit_to_check_msg)) {
                                    propagated_msg = soft_syndrome_magnitude;
                                    int check_node_sgn = sgn;
                                    if (check_nbr.bit_to_check_msg <= 0) {
                                        check_node_sgn ^= 1;
                                    }
                                    // now we check whether we have to update the soft syndrome magnitude and sign
                                    if (check_node_sgn == syndrome[check_index]) {
                                        if (std::abs(check_nbr.bit_to_check_msg) < min_bit_to_check_msg) {
                                            this->soft_syndrome[check_index] =
                                                    pow(-1, syndrome[check_index]) *
                                                    std::abs(check_nbr.bit_to_check_msg);
                                        } else {
                                            this->soft_syndrome[check_index] =
                                                    pow(-1, syndrome[check_index]) * min_bit_to_check_msg;
                                        }
                                    } else {
                                        syndrome[check_index] ^= 1;
                                        this->soft_syndrome[check_index] *= -1;
                                    }
                                }
                            }
                            sgn ^= syndrome[check_index];
                            check_nbr.check_to_bit_msg = ms_scaling_factor * pow(-1, sgn) * propagated_msg;
                            check_nbr.bit_to_check_msg = log_prob_ratios[bit_index];
                            log_prob_ratios[bit_index] += check_nbr.check_to_bit_msg;
                        }
                        // hard decision on bit
                        if (log_prob_ratios[bit_index] <= 0) {
                            decoding[bit_index] = 1;
                        } else {
                            decoding[bit_index] = 0;
                        }
                        temp = 0;
                        for (auto &e: pcm.reverse_iterate_column(bit_index)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }
                    // compute the syndrome for the current candidate decoding solution
                    loop_break = false;
                    CONVERGED = true;
                    for (auto i = 0; i < soft_info_syndrome.size(); i++) {
                        if (soft_info_syndrome[i] <= 0) {
                            candidate_syndrome[i] = 1;
                        } else {
                            candidate_syndrome[i] = 0;
                        }
                    }
                    candidate_syndrome = pcm.mulvec(decoding, candidate_syndrome);
                    for (auto i = 0; i < check_count && !loop_break; i++) {
                        if (candidate_syndrome[i] != syndrome[i]) {
                            CONVERGED = false;
                            loop_break = true;
                        }
                    }
                    iterations = it;
                }
                converge = CONVERGED;
                return decoding;
            }
        };
    }
}  // namespace ldpc::bp

#endif