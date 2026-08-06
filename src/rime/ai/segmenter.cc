#include "rime/ai/segmenter.hpp"

namespace aiime::lm {

namespace {

// Length in bytes of the UTF-8 character starting at data[0].
size_t Utf8CharLen(unsigned char first) {
  if (first < 0x80)
    return 1;
  if ((first & 0xE0) == 0xC0)
    return 2;
  if ((first & 0xF0) == 0xE0)
    return 3;
  if ((first & 0xF8) == 0xF0)
    return 4;
  return 1;
}

}  // namespace

std::vector<std::string> Segmenter::Segment(const std::string& text) const {
  std::vector<std::string> words;
  const size_t n = text.size();
  const size_t max_word = model_.MaxWordBytes();
  size_t i = 0;
  while (i < n) {
    size_t matched = 0;
    size_t probe_len = std::min(max_word, n - i);
    while (probe_len > 0) {
      if (model_.Id(text.substr(i, probe_len)) != Vocab::kUnknown) {
        matched = probe_len;
        break;
      }
      // step back by one whole UTF-8 character; guard against underflow
      // when the probe has shrunk below one character's length
      const size_t clen =
          Utf8CharLen(static_cast<unsigned char>(text[i + probe_len - 1]));
      probe_len = probe_len > clen ? probe_len - clen : 0;
    }
    if (matched > 0) {
      words.push_back(text.substr(i, matched));
      i += matched;
    } else {
      const size_t clen = Utf8CharLen(static_cast<unsigned char>(text[i]));
      words.push_back(text.substr(i, clen));
      i += clen;
    }
  }
  return words;
}

}  // namespace aiime::lm
