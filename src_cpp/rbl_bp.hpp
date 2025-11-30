#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace ldpc::rbl_bp {

class RBLBP_Decoder {
private:
    int N;
    int M;
    std::vector<std::vector<int>> H_rows;
    std::vector<std::vector<int>> H_cols;
    std::vector<std::vector<double>> R;
    std::vector<std::vector<double>> Q;

public:
    RBLBP_Decoder(int rows, int cols, const std::vector<std::vector<int>>& parity_matrix)
        : N(cols), M(rows) {
        H_rows.resize(M);
        H_cols.resize(N);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                if (parity_matrix[m][n] == 1) {
                    H_rows[m].push_back(n);
                    H_cols[n].push_back(m);
                }
            }
        }

        R.resize(M, std::vector<double>(N, 0.0));
        Q.resize(M, std::vector<double>(N, 0.0));
    }

    std::vector<int> decode(const std::vector<double>& received_LLR, int max_iter, double alpha) {
        for (auto& row : R) {
            std::fill(row.begin(), row.end(), 0.0);
        }

        for (int m = 0; m < M; ++m) {
            for (int n : H_rows[m]) {
                Q[m][n] = received_LLR[n];
            }
        }

        double min_abs_llr = std::numeric_limits<double>::max();
        for (double val : received_LLR) {
            if (std::abs(val) < min_abs_llr) {
                min_abs_llr = std::abs(val);
            }
        }

        double L_value_threshold = min_abs_llr;
        std::vector<int> decoded_bits(N);

        for (int l = 1; l <= max_iter; ++l) {
            std::vector<int> active_vars;
            for (int n = 0; n < N; ++n) {
                if (std::abs(received_LLR[n]) <= L_value_threshold) {
                    active_vars.push_back(n);
                }
            }

            for (int n : active_vars) {
                for (int k : H_cols[n]) {
                    for (int a : H_rows[k]) {
                        double product_msg = 1.0;
                        for (int n_prime : H_rows[k]) {
                            if (n_prime != a) {
                                const double t = std::tanh(Q[k][n_prime] / 2.0);
                                product_msg *= t;
                            }
                        }

                        if (product_msg > 0.999999) {
                            product_msg = 0.999999;
                        }
                        if (product_msg < -0.999999) {
                            product_msg = -0.999999;
                        }

                        R[k][a] = 2.0 * std::atanh(product_msg);
                    }

                    for (int a : H_rows[k]) {
                        for (int b : H_cols[a]) {
                            if (b == k) {
                                continue;
                            }

                            double sum_R = 0.0;
                            for (int m_prime : H_cols[a]) {
                                if (m_prime != b) {
                                    sum_R += R[m_prime][a];
                                }
                            }
                            Q[b][a] = received_LLR[a] + sum_R;
                        }
                    }
                }
            }

            L_value_threshold += alpha;

            bool valid_codeword = true;
            for (int n = 0; n < N; ++n) {
                double L_posterior = received_LLR[n];
                for (int m : H_cols[n]) {
                    L_posterior += R[m][n];
                }
                decoded_bits[n] = (L_posterior < 0) ? 1 : 0;
            }

            for (int m = 0; m < M; ++m) {
                int parity = 0;
                for (int n : H_rows[m]) {
                    parity ^= decoded_bits[n];
                }
                if (parity != 0) {
                    valid_codeword = false;
                    break;
                }
            }

            if (valid_codeword) {
                return decoded_bits;
            }
        }

        return decoded_bits;
    }
};

}  // namespace ldpc::rbl_bp
