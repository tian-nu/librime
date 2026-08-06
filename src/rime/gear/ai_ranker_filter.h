//
// AI candidate ranker — the core feature of this IME.
//
// Reorders the pinyin candidate list by the probability of each candidate
// being the next word after the committed context, as computed by the
// bundled Kneser-Ney n-gram language model:
//
//   P(w | 前面的词) — candidates are listed front to back by probability.
//
// Three modes, switched by context options (radio group in the schema):
//   * hybrid (ai_hybrid): the first `head` candidates are chosen by the LM,
//     the rest keep their traditional order;
//   * pure AI (ai_pure): every candidate is ranked by the LM;
//   * traditional (neither option set): this filter does not intervene.
//
#ifndef RIME_AI_RANKER_FILTER_H_
#define RIME_AI_RANKER_FILTER_H_

#include <memory>
#include <string>
#include <vector>

#include <rime/filter.h>
#include <rime/translation.h>

#include "rime/ai/ngram_model.hpp"
#include "rime/ai/segmenter.hpp"

namespace rime {

class AiRankerFilter : public Filter {
 public:
  explicit AiRankerFilter(const Ticket& ticket);

  an<Translation> Apply(an<Translation> translation,
                        CandidateList* candidates) override;

 private:
  bool LoadModel(const std::string& path);
  std::vector<std::string> ContextWords() const;
  std::vector<int32_t> ContextIds() const;
  double Score(const std::string& text, const std::vector<int32_t>& ctx) const;
  std::vector<an<Candidate>> Rerank(const CandidateList& pool, bool pure) const;

  bool enabled_ = false;
  int head_ = 3;
  std::string model_path_;
  std::string option_prefix_ = "ai_";
  std::unique_ptr<aiime::lm::NGramModel> model_;
  std::unique_ptr<aiime::lm::Segmenter> segmenter_;
};

}  // namespace rime

#endif  // RIME_AI_RANKER_FILTER_H_
