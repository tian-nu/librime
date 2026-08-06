#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aiime::lm {

// Token vocabulary. Id 0 is reserved for <unk> (unknown or pruned words).
class Vocab {
 public:
  static constexpr int32_t kUnknown = 0;

  Vocab() {
    word2id_.reserve(1 << 16);
    Add("<unk>");
  }

  int32_t Id(const std::string& word) const {
    auto it = word2id_.find(word);
    return it == word2id_.end() ? kUnknown : it->second;
  }

  // Training-time: assigns a new id to unseen words.
  int32_t Add(const std::string& word) {
    auto it = word2id_.find(word);
    if (it != word2id_.end())
      return it->second;
    const int32_t id = static_cast<int32_t>(id2word_.size());
    word2id_.emplace(word, id);
    id2word_.push_back(word);
    if (word.size() > max_word_bytes_)
      max_word_bytes_ = word.size();
    return id;
  }

  const std::string& Word(int32_t id) const {
    return id2word_.at(static_cast<size_t>(id));
  }

  size_t size() const { return id2word_.size(); }

  // Longest word in bytes; used to bound forward-maximum-matching.
  size_t MaxWordBytes() const { return max_word_bytes_; }

  void Save(std::ostream& os) const;
  void Load(std::istream& is);

 private:
  std::unordered_map<std::string, int32_t> word2id_;
  std::vector<std::string> id2word_;
  size_t max_word_bytes_ = 0;
};

}  // namespace aiime::lm
