#include "rime/gear/ai_ranker_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>

#include <rime/common.h>
#include <rime/config.h>
#include <rime/context.h>
#include <rime/schema.h>
#include <rime/ticket.h>

namespace rime {

namespace {

constexpr double kMinLogP = -100.0;  // floor for degenerate probabilities
constexpr double kUnknownWordP = 1e-6;

}  // namespace

AiRankerFilter::AiRankerFilter(const Ticket& ticket) : Filter(ticket) {
  if (!ticket.schema)
    return;
  Config* config = ticket.schema->config();
  if (!config)
    return;
  if (!config->GetBool("ai_ranker/enabled", &enabled_) || !enabled_)
    return;
  config->GetInt("ai_ranker/head", &head_);
  config->GetString("ai_ranker/model", &model_path_);
  config->GetString("ai_ranker/option_prefix", &option_prefix_);
  if (model_path_.empty())
    return;
  LoadModel(model_path_);
}

bool AiRankerFilter::LoadModel(const std::string& path) {
  std::string full = path;
  // Expand %VAR% environment variables (e.g. %APPDATA%\Rime\ai\model.lmbin).
  size_t i = 0;
  while (i < full.size()) {
    if (full[i] == '%') {
      const size_t j = full.find('%', i + 1);
      if (j != std::string::npos) {
        const std::string name = full.substr(i + 1, j - i - 1);
        const char* value = std::getenv(name.c_str());
        full.replace(i, j - i + 1, value ? value : "");
        i += value ? std::strlen(value) : 0;
        continue;
      }
    }
    ++i;
  }

  std::ifstream in(full, std::ios::binary);
  if (!in) {
    LOG(ERROR) << "ai_ranker: cannot open model file: " << full;
    return false;
  }
  auto model = std::make_unique<aiime::lm::NGramModel>();
  try {
    model->Load(in);
  } catch (const std::exception& e) {
    LOG(ERROR) << "ai_ranker: failed to load model: " << e.what();
    return false;
  }
  model_ = std::move(model);
  segmenter_ = std::make_unique<aiime::lm::Segmenter>(*model_);
  LOG(INFO) << "ai_ranker: model loaded, vocab=" << model_->vocab_size()
            << " contexts=" << model_->context_count()
            << " order=" << model_->order();
  return true;
}

an<Translation> AiRankerFilter::Apply(an<Translation> translation,
                                      CandidateList* candidates) {
  if (!enabled_ || !model_ || !segmenter_)
    return translation;
  Context* ctx = engine_->context();
  const bool pure = ctx->get_option(option_prefix_ + "pure");
  const bool hybrid = ctx->get_option(option_prefix_ + "hybrid");
  if (!pure && !hybrid)
    return translation;  // traditional mode: stay out of the way

  CandidateList pool;
  while (!translation->exhausted()) {
    auto candidate = translation->Peek();
    if (candidate)
      pool.push_back(candidate);
    translation->Next();
  }
  if (pool.empty())
    return translation;

  const auto reordered = Rerank(pool, pure);
  auto out = New<FifoTranslation>();
  for (const auto& candidate : reordered)
    out->Append(candidate);
  return out;
}

std::vector<std::string> AiRankerFilter::ContextWords() const {
  std::vector<std::string> words;
  const size_t max_ctx = static_cast<size_t>(std::max(1, model_->order() - 1));
  const auto& history = engine_->context()->commit_history();
  for (auto it = history.rbegin(); it != history.rend(); ++it) {
    if (it->type == "thru")
      continue;
    const auto tokens = segmenter_->Segment(it->text);
    for (auto t = tokens.rbegin(); t != tokens.rend() && words.size() < max_ctx;
         ++t)
      words.push_back(*t);
    if (words.size() >= max_ctx)
      break;
  }
  std::reverse(words.begin(), words.end());  // chronological order
  return words;
}

std::vector<int32_t> AiRankerFilter::ContextIds() const {
  const auto words = ContextWords();
  std::vector<int32_t> ids;
  ids.reserve(words.size());
  for (const auto& word : words)
    ids.push_back(model_->Id(word));
  return ids;
}

double AiRankerFilter::Score(const std::string& text,
                             const std::vector<int32_t>& ctx) const {
  const auto words = segmenter_->Segment(text);
  if (words.empty())
    return kMinLogP;
  std::vector<int32_t> window = ctx;
  const size_t max_window =
      static_cast<size_t>(std::max(1, model_->order() - 1));
  if (window.size() > max_window)
    window.erase(window.begin(), window.end() - max_window);

  double log_p = 0.0;
  for (const auto& word : words) {
    const int32_t id = model_->Id(word);
    double p = kUnknownWordP;
    if (id != aiime::lm::Vocab::kUnknown) {
      p = model_->P(id, window);
      if (p <= 0.0)
        p = kUnknownWordP;
    }
    log_p += std::log(p);
    window.push_back(id);
    if (window.size() > max_window)
      window.erase(window.begin());
  }
  // Normalize by word count so longer candidates are not systematically
  // outranked by one-word candidates.
  return std::max(log_p / static_cast<double>(words.size()), kMinLogP);
}

std::vector<an<Candidate>> AiRankerFilter::Rerank(const CandidateList& pool,
                                                  bool pure) const {
  const auto ctx = ContextIds();
  std::vector<std::pair<double, size_t>> scored;
  scored.reserve(pool.size());
  for (size_t i = 0; i < pool.size(); ++i)
    scored.emplace_back(Score(pool[i]->text(), ctx), i);
  std::stable_sort(
      scored.begin(), scored.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<an<Candidate>> out;
  out.reserve(pool.size());
  const int ai_count = pure ? static_cast<int>(scored.size())
                            : std::min(head_, static_cast<int>(scored.size()));
  if (ai_count <= 0) {
    for (const auto& candidate : pool)
      out.push_back(candidate);
    return out;
  }
  std::vector<bool> used(pool.size(), false);
  for (int k = 0; k < ai_count; ++k) {
    out.push_back(pool[scored[k].second]);
    used[scored[k].second] = true;
  }
  // the rest keep their traditional order
  for (size_t i = 0; i < pool.size(); ++i)
    if (!used[i])
      out.push_back(pool[i]);
  return out;
}

}  // namespace rime
