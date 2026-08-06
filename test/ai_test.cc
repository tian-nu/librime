//
// Tests for the AI candidate core: Kneser-Ney LM, forward-maximum-matching
// segmenter, and the context-aware candidate scoring used by AiRankerFilter.
//
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <rime/ai/ngram_model.hpp>
#include <rime/ai/segmenter.hpp>

using aiime::lm::NGramModel;
using aiime::lm::Segmenter;
using aiime::lm::Vocab;

namespace {

std::vector<std::vector<std::string>> SampleCorpus() {
  return {
      {"我", "今天", "去", "北京"},   {"我", "今天", "去", "上海"},
      {"我", "明天", "去", "北京"},   {"他", "今天", "去", "北京"},
      {"她", "昨天", "去", "广州"},   {"我们", "今天", "开会"},
      {"明天", "下午", "开会"},       {"下午", "三点", "开会"},
      {"会议", "在", "明天", "下午"}, {"北京", "的", "天气", "很", "好"},
      {"上海", "的", "天气", "也", "很", "好"}, {"我", "喜欢", "北京", "的", "胡同"},
      {"他", "喜欢", "上海", "的", "外滩"},     {"今天", "的", "天气", "很", "冷"},
  };
}

// Mirror of AiRankerFilter::Score: sequence log-probability of the candidate
// text under the committed context, normalized by word count.
double Score(const NGramModel& m, const Segmenter& seg, const std::string& text,
             const std::vector<int32_t>& ctx) {
  const auto words = seg.Segment(text);
  if (words.empty())
    return -100.0;
  std::vector<int32_t> window = ctx;
  const size_t max_window = static_cast<size_t>(m.order() - 1);
  if (window.size() > max_window)
    window.erase(window.begin(), window.end() - max_window);
  double log_p = 0.0;
  for (const auto& word : words) {
    const int32_t id = m.Id(word);
    double p = 1e-6;
    if (id != Vocab::kUnknown) {
      p = m.P(id, window);
      if (p <= 0.0)
        p = 1e-6;
    }
    log_p += std::log(p);
    window.push_back(id);
    if (window.size() > max_window)
      window.erase(window.begin());
  }
  return std::max(log_p / static_cast<double>(words.size()), -100.0);
}

}  // namespace

TEST(AiLm, Ranking) {
  NGramModel m;
  m.Train(SampleCorpus());
  EXPECT_EQ(m.TopK({"我", "今天"}, 1)[0].word, m.Id("去"));
  EXPECT_EQ(m.TopK({"我", "今天", "去"}, 1)[0].word, m.Id("北京"));
  EXPECT_EQ(m.TopK({"他", "明天"}, 1)[0].word, m.Id("下午"));  // backs off to 明天->下午
  EXPECT_EQ(m.TopK({"我", "猫", "今天"}, 1)[0].word, m.Id("去"));  // unknown word truncates
}

TEST(AiLm, ProbabilityInvariants) {
  NGramModel m;
  m.Train(SampleCorpus());
  const auto cands = m.TopK({"我", "今天", "去"}, 10);
  ASSERT_FALSE(cands.empty());
  double sum = 0.0;
  for (size_t i = 0; i < cands.size(); ++i) {
    EXPECT_GE(cands[i].prob, 0.0);
    EXPECT_LE(cands[i].prob, 1.0);
    if (i > 0)
      EXPECT_GE(cands[i - 1].prob, cands[i].prob);
    sum += cands[i].prob;
  }
  EXPECT_LE(sum, 1.0 + 1e-9);
}

TEST(AiLm, SaveLoadRoundtrip) {
  NGramModel m;
  m.Train(SampleCorpus());
  std::ostringstream buf(std::ios::binary);
  m.Save(buf);
  std::istringstream in(buf.str(), std::ios::binary);
  NGramModel m2;
  m2.Load(in);
  EXPECT_EQ(m2.order(), m.order());
  EXPECT_EQ(m2.vocab_size(), m.vocab_size());
  EXPECT_EQ(m2.context_count(), m.context_count());
  const auto a = m.TopK({"我", "今天", "去"}, 10);
  const auto b = m2.TopK({"我", "今天", "去"}, 10);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].word, b[i].word);
    EXPECT_DOUBLE_EQ(a[i].prob, b[i].prob);
  }
}

TEST(AiSegmenter, ForwardMaxMatching) {
  NGramModel m;
  m.Train(SampleCorpus());
  Segmenter seg(m);
  EXPECT_EQ(seg.Segment("我们今天去北京"),
            (std::vector<std::string>{"我们", "今天", "去", "北京"}));
  EXPECT_EQ(seg.Segment("上海天气很好"),
            (std::vector<std::string>{"上海", "天气", "很", "好"}));
  // unknown content falls back to single UTF-8 characters
  EXPECT_EQ(seg.Segment("abc北京"),
            (std::vector<std::string>{"a", "b", "c", "北京"}));
}

TEST(AiScore, ContextAwareOrdering) {
  NGramModel m;
  m.Train(SampleCorpus());
  Segmenter seg(m);
  const std::vector<int32_t> ctx = {m.Id("我"), m.Id("今天")};
  const double s_qu = Score(m, seg, "去", ctx);
  const double s_hui = Score(m, seg, "开会", ctx);
  const double s_beijing = Score(m, seg, "北京", ctx);
  EXPECT_GT(s_qu, s_hui);
  EXPECT_GT(s_qu, s_beijing);
  // length normalization: 去北京 is a plausible phrase, not crushed by 去
  const double s_qu_beijing = Score(m, seg, "去北京", ctx);
  EXPECT_GT(s_qu_beijing, s_beijing);
  EXPECT_GT(s_qu, s_qu_beijing);
}
