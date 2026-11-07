// Native execution of the reviewed Frozen65kTokenizer rules. Vocabulary and
// segmentation are inputs, never inferred from a stock Hugging Face export.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;
using Id = uint16_t;
using Ids = std::vector<Id>;
constexpr int BYTE = 32, FALLBACK = 288, INDIC = 544, TAIL = 40347, VOCAB = 65536;
const std::string MARK = "\xC4\xA0";

struct Rune { uint32_t value; size_t begin, end; };
std::vector<Rune> runes(const std::string& s) {
    std::vector<Rune> out; out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        size_t start = i; uint8_t first = static_cast<uint8_t>(s[i++]);
        uint32_t value; unsigned more;
        if (first < 128) { value = first; more = 0; }
        else if (first >= 0xC2 && first <= 0xDF) { value = first & 31; more = 1; }
        else if (first >= 0xE0 && first <= 0xEF) { value = first & 15; more = 2; }
        else if (first >= 0xF0 && first <= 0xF4) { value = first & 7; more = 3; }
        else throw std::invalid_argument("invalid UTF-8 input");
        if (i + more > s.size()) throw std::invalid_argument("truncated UTF-8 input");
        for (unsigned j = 0; j < more; ++j) {
            uint8_t c = static_cast<uint8_t>(s[i++]);
            if ((c & 0xC0) != 0x80) throw std::invalid_argument("invalid UTF-8 continuation");
            value = (value << 6) | (c & 63);
        }
        if ((more == 1 && value < 128) || (more == 2 && value < 2048) ||
            (more == 3 && value < 65536) || value > 0x10FFFF ||
            (value >= 0xD800 && value <= 0xDFFF))
            throw std::invalid_argument("noncanonical UTF-8 input");
        out.push_back({value, start, i});
    }
    return out;
}
bool indic(uint32_t c) {
    return (c >= 0x900 && c <= 0x9ff) || (c >= 0xa80 && c <= 0xaff) ||
           (c >= 0xb80 && c <= 0xbff) || (c >= 0xc00 && c <= 0xcff);
}
bool format(uint32_t c) { return c == 0x200c || c == 0x200d; }
bool protected_char(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
uint32_t pair_key(Id a, Id b) { return (uint32_t(a) << 16) | b; }
int decode_class(Id i) {
    if (i >= FALLBACK && i < TAIL) return 1;
    if (i >= BYTE && i < FALLBACK) {
        int b = i - BYTE;
        return b >= 128 || b == 9 || b == 10 || b == 13 ||
               (b >= 65 && b <= 90) || (b >= 97 && b <= 122) ? 0 : 2;
    }
    return 0;
}

// Cache memory is bounded by bytes, including a conservative per-entry charge.
class Cache {
    struct Entry { Ids ids; std::list<std::string>::iterator order; size_t charge; };
    std::unordered_map<std::string, Entry> entries;
    std::list<std::string> order;
    size_t used = 0, limit;
public:
    explicit Cache(size_t n) : limit(n) {}
    const Ids* get(const std::string& key) {
        auto at = entries.find(key); if (at == entries.end()) return nullptr;
        order.splice(order.begin(), order, at->second.order); return &at->second.ids;
    }
    void put(const std::string& key, const Ids& ids) {
        if (key.size() > 4096 || !limit || entries.count(key)) return;
        size_t charge = 2 * key.size() + 2 * ids.size() + 256;
        if (charge > limit) return;
        while (used + charge > limit && !order.empty()) {
            auto at = entries.find(order.back()); used -= at->second.charge;
            entries.erase(at); order.pop_back();
        }
        order.push_front(key); entries.emplace(key, Entry{ids, order.begin(), charge}); used += charge;
    }
    size_t bytes() const { return used; }
};

struct Merge { Id target; double priority; };
struct Candidate { double priority; int left, right; Id target, a, b; };
struct Later {
    bool operator()(const Candidate& a, const Candidate& b) const {
        if (a.priority != b.priority) return a.priority > b.priority;
        if (a.left != b.left) return a.left > b.left;
        if (a.right != b.right) return a.right > b.right;
        return a.target > b.target;
    }
};
struct Unit { int kind; std::string text; }; // 0 English, 1 Indic, 2 single space

class NativeEncoder {
    std::vector<std::string> pieces, decoded;
    std::vector<bool> dummy_prefix, reserved;
    std::unordered_map<std::string, Id> piece_id;
    std::unordered_map<uint32_t, Id> indic_atom;
    std::unordered_map<uint32_t, Merge> english_pairs, indic_pairs;
    std::vector<uint8_t> latin, properties;
    std::vector<std::pair<std::string, Id>> seeds;
    py::object scissors;
    Cache english_cache, indic_cache;

    Ids bpe(Ids ids, const std::unordered_map<uint32_t, Merge>& pairs, bool restrict_space) const {
        int n = static_cast<int>(ids.size()); if (n < 2) return ids;
        std::vector<int> prev(n), next(n); std::vector<uint8_t> alive(n, 1);
        int first_non_neutral = n;
        for (int i = 0; i < n; ++i) {
            prev[i] = i - 1; next[i] = i + 1;
            if (ids[i] < BYTE || ids[i] >= FALLBACK) first_non_neutral = std::min(first_non_neutral, i);
        }
        std::priority_queue<Candidate, std::vector<Candidate>, Later> heap;
        auto restricted = [&](int left) { return restrict_space && ids[left] == 64 && first_non_neutral >= left; };
        auto offer = [&](int left) {
            if (left < 0 || !alive[left]) return;
            int right = next[left]; if (right >= n || !alive[right] || restricted(left)) return;
            auto at = pairs.find(pair_key(ids[left], ids[right])); if (at == pairs.end()) return;
            heap.push({at->second.priority, left, right, at->second.target, ids[left], ids[right]});
        };
        for (int i = 0; i < n - 1; ++i) offer(i);
        while (!heap.empty()) {
            auto c = heap.top(); heap.pop(); int l = c.left, r = c.right;
            if (!alive[l] || !alive[r] || next[l] != r || ids[l] != c.a || ids[r] != c.b || restricted(l)) continue;
            alive[r] = 0; ids[l] = c.target; next[l] = next[r];
            if (next[r] < n) prev[next[r]] = l;
            // Every possible scored multi-character merge targets a non-byte
            // piece. Thus the earliest non-neutral position only moves left.
            if (ids[l] < BYTE || ids[l] >= FALLBACK) first_non_neutral = std::min(first_non_neutral, l);
            offer(l); offer(prev[l]);
        }
        Ids out; out.reserve(n); for (int i = 0; i < n; ++i) if (alive[i]) out.push_back(ids[i]); return out;
    }

    Ids english_word(const std::string& word) {
        if (const auto* found = english_cache.get(word)) return *found;
        Ids ids; ids.reserve(word.size()); size_t offset = 0;
        while (offset < word.size() && (protected_char(word[offset]) || word[offset] == '\\')) {
            bool matched = false;
            for (const auto& seed : seeds) {
                if (word.compare(offset, seed.first.size(), seed.first) == 0) {
                    ids.push_back(seed.second); offset += seed.first.size(); matched = true; break;
                }
            }
            if (!matched) break;
        }
        for (; offset < word.size(); ++offset) ids.push_back(BYTE + static_cast<uint8_t>(word[offset]));
        ids = bpe(std::move(ids), english_pairs, false); english_cache.put(word, ids); return ids;
    }
    Ids indic_body(const std::string& body, bool with_dummy, bool restrict_space) const {
        Ids ids; if (with_dummy) ids.push_back(64);
        for (const auto& c : runes(body)) {
            if (c.value == 32 || c.value == 0x120) ids.push_back(64);
            else if (auto at = indic_atom.find(c.value); at != indic_atom.end()) ids.push_back(at->second);
            else if (c.value < 128) ids.push_back(BYTE + c.value);
            else for (size_t j = c.begin; j < c.end; ++j) ids.push_back(FALLBACK + static_cast<uint8_t>(body[j]));
        }
        return bpe(std::move(ids), indic_pairs, restrict_space);
    }
    Ids indic_span(const std::string& body, bool with_dummy) {
        std::string key = std::string(1, with_dummy ? '1' : '0') + body;
        if (const auto* found = indic_cache.get(key)) return *found;
        Ids ids = indic_body(body, with_dummy, !with_dummy);
        if (with_dummy && ids.size() > 1 && ids[0] == 64) ids = indic_body(body, false, true);
        indic_cache.put(key, ids); return ids;
    }

    std::vector<std::string> split_english(const std::string& text) const {
        // Exact ordered alternatives of the frozen english_scissors pattern.
        // Unicode property tables come from the pinned regex library, so this
        // removes per-word Python objects without substituting Unicode rules.
        auto chars = runes(text); size_t n = chars.size();
        std::vector<std::string> words; words.reserve(n / 4 + 1);
        auto cp = [&](size_t i) { return chars[i].value; };
        auto prop = [&](size_t i) { return properties[cp(i)]; };
        auto lm = [&](size_t i) { return (prop(i) & 3) != 0; };
        auto ascii_letter = [&](size_t i) { auto c = cp(i); return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
        size_t i = 0;
        while (i < n) {
            size_t end = i; auto c = cp(i);
            if (c == '\r' || c == '\n') {
                end = i + 1;
                if (c == '\r' && end < n && cp(end) == '\n') ++end;
                while (end < n && (cp(end) == ' ' || cp(end) == '\t')) ++end;
            } else if (c == '\\' && i + 1 < n && ascii_letter(i + 1)) {
                end = i + 2; while (end < n && ascii_letter(end)) ++end;
            } else if (c >= '0' && c <= '9') {
                end = i + 1;
            } else {
                bool prefix = c != '\r' && c != '\n' && !(prop(i) & (1 | 4));
                if (prefix && i + 1 < n && lm(i + 1)) end = i + 1;
                // The optional prefix backtracks when it consumes the only
                // combining mark; the mandatory L/M run can then start at i.
                if ((end > i && lm(end)) || (end == i && lm(i))) {
                    while (end < n && lm(end)) ++end;
                    while (end + 1 < n && (cp(end) == '\'' || cp(end) == 0x2019) && lm(end + 1)) {
                        end += 2; while (end < n && lm(end)) ++end;
                    }
                } else if (!(prop(i) & (8 | 1 | 2 | 4))) {
                    end = i + 1; while (end < n && !(prop(end) & (8 | 1 | 2 | 4))) ++end;
                } else if ((prop(i) & 8) && c != '\r' && c != '\n') {
                    end = i + 1;
                    while (end < n && (prop(end) & 8) && cp(end) != '\r' && cp(end) != '\n') ++end;
                } else end = i + 1;
            }
            size_t begin_byte = chars[i].begin, end_byte = chars[end - 1].end;
            words.push_back(text.substr(begin_byte, end_byte - begin_byte)); i = end;
        }
        return words;
    }

    void route_stretch(const std::string& s, std::vector<Unit>& out) const {
        // All kept Indic blocks have UTF-8 lead byte E0. Format characters
        // cannot create an Indic span without a kept Indic neighbor.
        if (s.find('\xE0') == std::string::npos) { out.push_back({0, s}); return; }
        auto chars = runes(s); size_t n = chars.size();
        std::vector<uint8_t> classes(n, 0), right_indic(n, 0); bool next_indic = false;
        for (size_t i = n; i-- > 0;) {
            right_indic[i] = next_indic;
            if (!format(chars[i].value)) next_indic = indic(chars[i].value);
        }
        bool left_indic = false, has_i = false, has_l = false;
        for (size_t i = 0; i < n; ++i) {
            auto c = chars[i].value;
            if (latin[c]) classes[i] = 2;
            else if (indic(c) || (format(c) && (left_indic || right_indic[i]))) classes[i] = 1;
            if (!format(c)) left_indic = indic(c);
            has_i |= classes[i] == 1; has_l |= classes[i] == 2;
        }
        if (!has_i || !has_l) { out.push_back({has_i ? 1 : 0, s}); return; }
        std::vector<Unit> units; size_t start = 0;
        for (size_t end = 0; end <= n; ++end) {
            if (end < n && chars[end].value != 32) continue;
            if (start > 0) units.push_back({2, " "});
            std::vector<Unit> runs;
            for (size_t i = start; i < end; ++i) {
                int kind = classes[i] == 1 ? 1 : (classes[i] == 2 ? 0 : -1);
                std::string piece = s.substr(chars[i].begin, chars[i].end - chars[i].begin);
                if (kind == -1) {
                    if (runs.empty()) runs.push_back({-1, piece}); else runs.back().text += piece;
                } else {
                    if (!runs.empty() && runs.back().kind == -1) runs.back().kind = kind;
                    if (!runs.empty() && runs.back().kind == kind) runs.back().text += piece;
                    else runs.push_back({kind, piece});
                }
            }
            if (!runs.empty() && runs.front().kind == -1) runs.front().kind = 0;
            for (auto& u : runs) units.push_back(std::move(u));
            start = end + 1;
        }
        std::vector<Unit> merged;
        for (size_t i = 0; i < units.size(); ++i) {
            auto& u = units[i];
            if (u.kind == 2 && !merged.empty() && merged.back().kind == 1 && i + 1 < units.size() && units[i + 1].kind == 1)
                merged.back().text += u.text;
            else if (u.kind == 1 && !merged.empty() && merged.back().kind == 1) merged.back().text += u.text;
            else merged.push_back({u.kind == 2 ? 0 : u.kind, std::move(u.text)});
        }
        for (auto& u : merged) out.push_back(std::move(u));
    }
    std::vector<Unit> route(const std::string& s) const {
        std::vector<Unit> out; size_t start = 0, i = 0;
        while (i < s.size()) {
            if (!protected_char(s[i])) { ++i; continue; }
            size_t end = i + 1; while (end < s.size() && protected_char(s[end])) ++end;
            if (!(end == i + 1 && s[i] == ' ')) {
                if (i > start) route_stretch(s.substr(start, i - start), out);
                out.push_back({0, s.substr(i, end - i)}); start = end;
            }
            i = end;
        }
        if (start < s.size()) route_stretch(s.substr(start), out);
        return out;
    }

public:
    NativeEncoder(py::dict artifact, py::object scissors_, const std::vector<uint32_t>& latin_points,
                  const std::string& unicode_properties,
                  size_t english_cache_bytes, size_t indic_cache_bytes)
        : scissors(std::move(scissors_)), english_cache(english_cache_bytes), indic_cache(indic_cache_bytes) {
        pieces = artifact["pieces"].cast<std::vector<std::string>>();
        if (pieces.size() != VOCAB) throw std::invalid_argument("expected 65536 vocabulary entries");
        latin.assign(0x110000, 0); for (auto c : latin_points) {
            if (c >= latin.size()) throw std::invalid_argument("invalid Latin classification table"); latin[c] = 1;
        }
        if (unicode_properties.size() != 0x110000) throw std::invalid_argument("invalid regex Unicode property table");
        properties.assign(unicode_properties.begin(), unicode_properties.end());
        for (int i = 0; i < VOCAB; ++i) if (!piece_id.emplace(pieces[i], Id(i)).second) throw std::invalid_argument("duplicate piece");
        auto alphabet = artifact["bytes_to_unicode"].cast<std::vector<std::string>>();
        if (alphabet.size() != 256) throw std::invalid_argument("invalid byte alphabet");
        std::unordered_map<std::string, uint8_t> byte_of;
        for (int b = 0; b < 256; ++b) byte_of[alphabet[b]] = uint8_t(b);
        auto english_merges = artifact["english_merges"].cast<std::vector<std::pair<std::string, std::string>>>();
        for (size_t r = 0; r < english_merges.size(); ++r) {
            const auto& p = english_merges[r]; Id target = TAIL + r;
            if (r + TAIL >= VOCAB || p.first + p.second != pieces[target]) throw std::invalid_argument("invalid English merge order");
            english_pairs[pair_key(piece_id.at(p.first), piece_id.at(p.second))] = {target, double(r)};
        }
        py::list scores = artifact["indic_scores"].cast<py::list>();
        if (scores.size() != VOCAB) throw std::invalid_argument("invalid Indic scores");
        for (int i = 0; i < VOCAB; ++i) {
            if (scores[i].is_none()) continue;
            double score = py::cast<double>(scores[i]); if (!std::isfinite(score)) throw std::invalid_argument("nonfinite Indic score");
            auto chars = runes(pieces[i]);
            if (i >= INDIC && i < TAIL && chars.size() == 1) indic_atom[chars[0].value] = Id(i);
            for (size_t k = 1; k < chars.size(); ++k) {
                size_t split = chars[k].begin;
                auto l = piece_id.find(pieces[i].substr(0, split)), r = piece_id.find(pieces[i].substr(split));
                if (l != piece_id.end() && r != piece_id.end()) {
                    if (i < INDIC) throw std::invalid_argument("unexpected neutral multi-character merge target");
                    indic_pairs[pair_key(l->second, r->second)] = {Id(i), -score};
                }
            }
        }
        for (const auto& s : artifact["english_seeds"].cast<std::vector<std::string>>()) {
            std::string mapped; for (auto c : s) mapped += alphabet[static_cast<uint8_t>(c)];
            seeds.emplace_back(s, piece_id.at(mapped));
        }
        std::stable_sort(seeds.begin(), seeds.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        decoded.resize(VOCAB); dummy_prefix.resize(VOCAB); reserved.resize(VOCAB);
        for (int i = 0; i < VOCAB; ++i) {
            reserved[i] = i < 32 || pieces[i].rfind("<reserved_space_", 0) == 0;
            if (i >= BYTE && i < FALLBACK) decoded[i] = std::string(1, char(i - BYTE));
            else if (i >= FALLBACK && i < INDIC) decoded[i] = std::string(1, char(i - FALLBACK));
            else if (i >= TAIL) {
                for (const auto& c : runes(pieces[i])) decoded[i].push_back(char(byte_of.at(pieces[i].substr(c.begin, c.end - c.begin))));
            } else if (i >= INDIC) {
                dummy_prefix[i] = pieces[i].rfind(MARK, 0) == 0 && pieces[i].size() > MARK.size();
                for (const auto& c : runes(pieces[i])) decoded[i] += c.value == 0x120 ? " " : pieces[i].substr(c.begin, c.end - c.begin);
            } else decoded[i] = pieces[i];
        }
    }
    Ids encode(const std::string& text, bool verify) {
        Ids out; out.reserve(text.size() / 2); bool single_space_pending = false;
        for (auto& u : route(text)) {
            if (u.kind == 1) {
                if (!u.text.empty() && u.text[0] == ' ') {
                    out.push_back(64); u.text.erase(0, 1);
                    if (u.text.empty()) { single_space_pending = false; continue; }
                }
                bool dummy = true;
                if (single_space_pending) {
                    auto i = static_cast<ptrdiff_t>(out.size()) - 2;
                    while (i >= 0 && decode_class(out[i]) == 2) --i;
                    if (i >= 0 && decode_class(out[i]) == 1) dummy = false;
                }
                auto ids = indic_span(u.text, dummy); out.insert(out.end(), ids.begin(), ids.end()); single_space_pending = false;
            } else {
                for (const auto& word : split_english(u.text)) {
                    auto ids = english_word(word); out.insert(out.end(), ids.begin(), ids.end());
                }
                single_space_pending = u.text == " ";
            }
        }
        for (auto id : out) if (reserved[id]) throw std::invalid_argument("normal text emitted a special/reserved ID");
        if (verify && decode_bytes(out, false) != text) throw std::invalid_argument("native tokenizer did not round-trip normalized source text");
        return out;
    }
    std::string decode_bytes(const Ids& ids, bool render_specials) const {
        std::string text; text.reserve(ids.size() * 3); bool previous_indic = false;
        for (auto id : ids) {
            if (id < 32 && !render_specials) throw std::invalid_argument("special/reserved ID requires explicit rendering");
            if (id >= INDIC && id < TAIL && dummy_prefix[id] && !previous_indic) text.append(decoded[id], 1, std::string::npos);
            else text += decoded[id];
            int cls = decode_class(id); if (cls != 2) previous_indic = cls == 1;
        }
        return text;
    }
    py::bytes encode_bytes(const std::string& text, bool verify) {
        auto ids = encode(text, verify); std::string bytes; bytes.resize(ids.size() * 2);
        for (size_t i = 0; i < ids.size(); ++i) { bytes[2 * i] = char(ids[i] & 255); bytes[2 * i + 1] = char(ids[i] >> 8); }
        return py::bytes(bytes);
    }
    py::list encode_batch_bytes(const std::vector<std::string>& texts, bool verify) {
        py::list out; for (const auto& text : texts) out.append(encode_bytes(text, verify)); return out;
    }
    py::dict statistics() const {
        py::dict d; d["english_pairs"] = english_pairs.size(); d["indic_pairs"] = indic_pairs.size();
        d["english_cache_bytes"] = english_cache.bytes(); d["indic_cache_bytes"] = indic_cache.bytes(); return d;
    }
    std::vector<std::string> scissors_words(const std::string& text) const { return split_english(text); }
};
PYBIND11_MODULE(_terminus_tokenizer_native, m) {
    m.attr("ABI_VERSION") = 2;
    py::class_<NativeEncoder>(m, "NativeEncoder")
        .def(py::init<py::dict, py::object, const std::vector<uint32_t>&, const std::string&, size_t, size_t>())
        .def("encode", &NativeEncoder::encode, py::arg("normalized_text"), py::arg("verify") = true)
        .def("encode_bytes", &NativeEncoder::encode_bytes, py::arg("normalized_text"), py::arg("verify") = true)
        .def("encode_batch_bytes", &NativeEncoder::encode_batch_bytes, py::arg("normalized_texts"), py::arg("verify") = true)
        .def("decode_bytes", [](const NativeEncoder& self, const Ids& ids, bool render) { return py::bytes(self.decode_bytes(ids, render)); }, py::arg("ids"), py::arg("render_specials") = false)
        .def("statistics", &NativeEncoder::statistics)
        .def("scissors_words", &NativeEncoder::scissors_words);
}
