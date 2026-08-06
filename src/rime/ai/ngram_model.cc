#include "rime/ai/ngram_model.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace aiime::lm {

namespace {

// Context key: packed little-endian ids. Not portable across endianness,
// but consistent between training and querying on the same machine.
std::string CtxKey(const std::vector<int32_t>& ids) {
  std::string s;
  s.reserve(ids.size() * 4);
  for (int32_t id : ids)
    s.append(reinterpret_cast<const char*>(&id), sizeof(id));
  return s;
}

void WriteU32(std::ostream& os, uint32_t v) {
  os.write(reinterpret_cast<const char*>(&v), 4);
}
void WriteI64(std::ostream& os, int64_t v) {
  os.write(reinterpret_cast<const char*>(&v), 8);
}
void WriteF64(std::ostream& os, double v) {
  os.write(reinterpret_cast<const char*>(&v), 8);
}

uint32_t ReadU32(std::istream& is) {
  uint32_t v = 0;
  is.read(reinterpret_cast<char*>(&v), 4);
  return v;
}
int64_t ReadI64(std::istream& is) {
  int64_t v = 0;
  is.read(reinterpret_cast<char*>(&v), 8);
  return v;
}
double ReadF64(std::istream& is) {
  double v = 0;
  is.read(reinterpret_cast<char*>(&v), 8);
  return v;
}

void WriteI64s(std::ostream& os, const std::vector<int64_t>& v) {
  WriteU32(os, static_cast<uint32_t>(v.size()));
  for (int64_t x : v)
    WriteI64(os, x);
}

std::vector<int64_t> ReadI64s(std::istream& is) {
  const uint32_t n = ReadU32(is);
  std::vector<int64_t> v(n);
  for (uint32_t i = 0; i < n; ++i)
    v[i] = ReadI64(is);
  return v;
}

void CheckStream(std::istream& is, const char* what) {
  if (!is)
    throw std::runtime_error(std::string("corrupt or truncated model: ") +
                             what);
}

}  // namespace

void Vocab::Save(std::ostream& os) const {
  WriteU32(os, static_cast<uint32_t>(id2word_.size()));
  for (const auto& w : id2word_) {
    WriteU32(os, static_cast<uint32_t>(w.size()));
    os.write(w.data(), static_cast<std::streamsize>(w.size()));
  }
}

void Vocab::Load(std::istream& is) {
  word2id_.clear();
  id2word_.clear();
  Add("<unk>");  // re-register id 0 so the file's own "<unk>" dedups instead of
                 // shifting ids
  const uint32_t n = ReadU32(is);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t len = ReadU32(is);
    std::string w(len, '\0');
    is.read(w.data(), static_cast<std::streamsize>(len));
    CheckStream(is, "vocab");
    Add(w);
  }
}

void NGramModel::Train(const std::vector<std::vector<std::string>>& lines,
                       int32_t order,
                       int32_t min_count,
                       double discount) {
  order_ = std::max(1, order);
  discount_ = discount;

  // Pass 1: raw unigram counts decide the final vocabulary.
  std::unordered_map<std::string, int64_t> raw;
  for (const auto& line : lines)
    for (const auto& w : line)
      ++raw[w];

  // Pass 2: convert to ids, pruning rare words to <unk>.
  std::vector<std::vector<int32_t>> ids;
  ids.reserve(lines.size());
  for (const auto& line : lines) {
    std::vector<int32_t> l;
    l.reserve(line.size());
    for (const auto& w : line) {
      auto it = raw.find(w);
      l.push_back(it != raw.end() && it->second >= min_count ? vocab_.Add(w)
                                                             : Vocab::kUnknown);
    }
    ids.push_back(std::move(l));
  }

  unigram_.assign(vocab_.size(), 0);
  total_ = 0;
  for (const auto& l : ids)
    for (int32_t id : l) {
      ++unigram_[static_cast<size_t>(id)];
      ++total_;
    }

  follow_.clear();
  for (int32_t n = 2; n <= order_; ++n) {
    for (const auto& l : ids) {
      if (static_cast<int32_t>(l.size()) < n)
        continue;
      for (size_t i = 0; i + static_cast<size_t>(n) <= l.size(); ++i) {
        std::vector<int32_t> ctx(l.begin() + static_cast<ptrdiff_t>(i),
                                 l.begin() + static_cast<ptrdiff_t>(i + n - 1));
        ++follow_[CtxKey(ctx)][l[i + n - 1]];
      }
    }
  }

  continuation_.assign(vocab_.size(), 0);
  for (const auto& kv : follow_)
    if (kv.first.size() == sizeof(int32_t))  // order-2 contexts only
      for (const auto& fw : kv.second)
        ++continuation_[static_cast<size_t>(fw.first)];
  total_continuation_ = 0;
  for (int64_t c : continuation_)
    total_continuation_ += c;

  RebuildDerived();
}

double NGramModel::P1(int32_t word) const {
  if (total_continuation_ == 0)
    return 0.0;
  return static_cast<double>(continuation_.at(static_cast<size_t>(word))) /
         static_cast<double>(total_continuation_);
}

double NGramModel::P(int32_t word, const std::vector<int32_t>& context) const {
  double prob = P1(word);
  const int32_t max_len =
      std::min<int32_t>(order_ - 1, static_cast<int32_t>(context.size()));
  for (int32_t len = 1; len <= max_len; ++len) {
    std::vector<int32_t> ctx(context.end() - len, context.end());
    auto it = follow_.find(CtxKey(ctx));
    if (it == follow_.end())
      continue;  // unseen context: back off, prob unchanged
    const auto& f = it->second;
    int64_t c_h = 0;
    for (const auto& kv : f)
      c_h += kv.second;
    const auto wi = f.find(word);
    const double c_hw = wi == f.end() ? 0.0 : static_cast<double>(wi->second);
    const double alpha =
        discount_ * static_cast<double>(f.size()) / static_cast<double>(c_h);
    prob = std::max(c_hw - discount_, 0.0) / static_cast<double>(c_h) +
           alpha * prob;
  }
  return prob;
}

std::vector<NGramModel::Candidate> NGramModel::TopK(
    const std::vector<std::string>& context,
    int32_t k) const {
  k = std::max(1, k);
  // Longest suffix of known words acts as the context; an unknown word
  // truncates everything before it (backoff semantics).
  std::vector<int32_t> ids;
  for (auto it = context.rbegin();
       it != context.rend() && ids.size() < static_cast<size_t>(order_ - 1);
       ++it) {
    const int32_t id = vocab_.Id(*it);
    if (id == Vocab::kUnknown)
      break;
    ids.push_back(id);
  }
  std::reverse(ids.begin(), ids.end());

  // Candidate pool: every word following any context in the backoff chain,
  // padded with the top continuation words for coverage.
  std::vector<int32_t> pool;
  for (int32_t len = 1; len <= static_cast<int32_t>(ids.size()); ++len) {
    auto it =
        follow_.find(CtxKey(std::vector<int32_t>(ids.end() - len, ids.end())));
    if (it == follow_.end())
      continue;
    for (const auto& kv : it->second)
      pool.push_back(kv.first);
  }
  for (int32_t w : top_continuation_) {
    if (static_cast<int32_t>(pool.size()) >= k)
      break;
    pool.push_back(w);
  }
  std::sort(pool.begin(), pool.end());
  pool.erase(std::unique(pool.begin(), pool.end()), pool.end());

  std::vector<Candidate> out;
  out.reserve(pool.size());
  for (int32_t w : pool)
    out.push_back({w, P(w, ids)});
  std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
    return a.prob > b.prob;
  });
  if (static_cast<int32_t>(out.size()) > k)
    out.resize(static_cast<size_t>(k));
  return out;
}

void NGramModel::RebuildDerived() {
  top_continuation_.clear();
  top_continuation_.reserve(vocab_.size());
  for (int32_t w = 0; w < static_cast<int32_t>(vocab_.size()); ++w)
    if (continuation_[static_cast<size_t>(w)] > 0)
      top_continuation_.push_back(w);
  std::sort(top_continuation_.begin(), top_continuation_.end(),
            [this](int32_t a, int32_t b) { return P1(a) > P1(b); });
}

void NGramModel::Save(std::ostream& os) const {
  os.write("AILM", 4);
  WriteU32(os, 1);  // format version
  WriteU32(os, static_cast<uint32_t>(order_));
  WriteF64(os, discount_);
  WriteI64(os, total_);
  WriteI64(os, total_continuation_);
  vocab_.Save(os);
  WriteI64s(os, unigram_);
  WriteI64s(os, continuation_);
  WriteU32(os, static_cast<uint32_t>(follow_.size()));
  for (const auto& kv : follow_) {
    WriteU32(os, static_cast<uint32_t>(kv.first.size()));
    os.write(kv.first.data(), static_cast<std::streamsize>(kv.first.size()));
    WriteU32(os, static_cast<uint32_t>(kv.second.size()));
    for (const auto& fw : kv.second) {
      WriteU32(os, static_cast<uint32_t>(fw.first));
      WriteI64(os, fw.second);
    }
  }
}

void NGramModel::Load(std::istream& is) {
  char magic[4] = {};
  is.read(magic, 4);
  if (std::memcmp(magic, "AILM", 4) != 0)
    throw std::runtime_error("not an AILM model file");
  const uint32_t version = ReadU32(is);
  if (version != 1)
    throw std::runtime_error("unsupported model version");
  order_ = static_cast<int32_t>(ReadU32(is));
  discount_ = ReadF64(is);
  total_ = ReadI64(is);
  total_continuation_ = ReadI64(is);
  vocab_.Load(is);
  unigram_ = ReadI64s(is);
  continuation_ = ReadI64s(is);
  const uint32_t n_ctx = ReadU32(is);
  follow_.clear();
  follow_.reserve(n_ctx);
  for (uint32_t i = 0; i < n_ctx; ++i) {
    const uint32_t key_len = ReadU32(is);
    std::string key(key_len, '\0');
    is.read(key.data(), static_cast<std::streamsize>(key_len));
    const uint32_t n_follow = ReadU32(is);
    auto& f = follow_[std::move(key)];
    f.reserve(n_follow);
    for (uint32_t j = 0; j < n_follow; ++j) {
      const int32_t w = static_cast<int32_t>(ReadU32(is));
      f[w] = ReadI64(is);
    }
  }
  CheckStream(is, "follow table");
  RebuildDerived();
}

}  // namespace aiime::lm
