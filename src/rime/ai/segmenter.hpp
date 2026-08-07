#pragma once

#include <string>
#include <vector>

#include <rime_api.h>

#include "rime/ai/ngram_model.hpp"

namespace aiime::lm {

// Lightweight forward-maximum-matching segmenter backed by the model's own
// vocabulary: every token the LM knows can be matched, unknown text falls
// back to single UTF-8 characters.
class RIME_DLL Segmenter {
 public:
  explicit Segmenter(const NGramModel& model) : model_(model) {}

  // Splits continuous text (no spaces) into a word sequence.
  std::vector<std::string> Segment(const std::string& text) const;

 private:
  const NGramModel& model_;
};

}  // namespace aiime::lm
