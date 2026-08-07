#pragma once

#include "rime/ai/vocab.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

#include <rime_api.h>

namespace aiime::lm {

// Interpolated Kneser-Ney n-gram language model — the "AI candidate" core.
//
//   P(w | h) = max(c(h,w) - D, 0)/c(h)  +  D * N1+(h*) / c(h) * P(w |
//   suffix(h))
//
// with the unigram level being the continuation probability
//
//   P1(w) = N1+(*w) / N1+(**)
//
// so that a word unseen under a seen context still receives meaningful mass.
// This is exactly "compute the probability of the next word given the words
// before it" — candidates are ranked by P(w | context), front to back.
class RIME_DLL NGramModel {
 public:
  static constexpr int32_t kDefaultOrder = 3;
  static constexpr double kDefaultDiscount = 0.75;

  struct Candidate {
    int32_t word;
    double prob;
  };

  // Trains on lines of whitespace-separated tokens (pre-segmented text).
  void Train(const std::vector<std::vector<std::string>>& lines,
             int32_t order = kDefaultOrder,
             int32_t min_count = 1,
             double discount = kDefaultDiscount);

  // Next-word prediction: top-k words under the context, P(w|context)
  // descending.
  std::vector<Candidate> TopK(const std::vector<std::string>& context,
                              int32_t k) const;

  double P(int32_t word, const std::vector<int32_t>& context) const;
  double P1(int32_t word) const;  // continuation probability

  int32_t Id(const std::string& word) const { return vocab_.Id(word); }
  const std::string& Word(int32_t id) const { return vocab_.Word(id); }
  size_t MaxWordBytes() const { return vocab_.MaxWordBytes(); }

  int32_t order() const { return order_; }
  double discount() const { return discount_; }
  size_t vocab_size() const { return vocab_.size(); }
  int64_t total_tokens() const { return total_; }
  size_t context_count() const { return follow_.size(); }

  void Save(std::ostream& os) const;
  void Load(std::istream& is);

 private:
  void RebuildDerived();

  int32_t order_ = kDefaultOrder;
  double discount_ = kDefaultDiscount;
  Vocab vocab_;
  std::vector<int64_t> unigram_;       // c(w)
  std::vector<int64_t> continuation_;  // distinct predecessors of w
  int64_t total_ = 0;                  // sum of c(w)
  int64_t total_continuation_ = 0;     // number of distinct bigram types
  std::unordered_map<std::string, std::unordered_map<int32_t, int64_t>> follow_;
  std::vector<int32_t> top_continuation_;  // words by P1, descending
};

}  // namespace aiime::lm
