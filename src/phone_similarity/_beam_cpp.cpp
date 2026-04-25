#include <Python.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

struct Node {
  int parent;
  int entry;
  int consumed;
  int depth;
  double raw;
  double score;
};

struct RescoreResult {
  double distance;
  std::vector<long> entry_ids;
  double raw_cost;
};

double dp_edit_distance(
    const Py_ssize_t* src,
    const Py_ssize_t src_len,
    const Py_ssize_t* tgt,
    const Py_ssize_t tgt_len,
    const double* dist_flat,
    const Py_ssize_t dim) {
  std::vector<double> prev(static_cast<size_t>(tgt_len + 1));
  std::vector<double> curr(static_cast<size_t>(tgt_len + 1));

  for (Py_ssize_t j = 0; j <= tgt_len; ++j) {
    prev[static_cast<size_t>(j)] = static_cast<double>(j);
  }

  const Py_ssize_t unk = dim - 1;
  for (Py_ssize_t i = 1; i <= src_len; ++i) {
    curr[0] = static_cast<double>(i);

    Py_ssize_t s = src[i - 1];
    if (s < 0 || s >= dim) {
      s = unk;
    }

    const double* row = dist_flat + (s * dim);
    for (Py_ssize_t j = 1; j <= tgt_len; ++j) {
      Py_ssize_t t = tgt[j - 1];
      if (t < 0 || t >= dim) {
        t = unk;
      }

      const double sub = row[t];
      const double del_cost = prev[static_cast<size_t>(j)] + 1.0;
      const double ins_cost = curr[static_cast<size_t>(j - 1)] + 1.0;
      const double sub_cost = prev[static_cast<size_t>(j - 1)] + sub;
      curr[static_cast<size_t>(j)] = std::min(del_cost, std::min(ins_cost, sub_cost));
    }
    prev.swap(curr);
  }

  return prev[static_cast<size_t>(tgt_len)];
}

PyObject* beam_state_search_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
  PyObject* candidates_by_consumed = nullptr;
  int source_len = 0;
  int beam_width = 10;
  int max_words = 4;
  double max_distance = 0.50;
  double prune_ratio = 2.0;

  static const char* kwlist[] = {
      "candidates_by_consumed", "source_len", "beam_width", "max_words", "max_distance",
      "prune_ratio", nullptr};

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "Oi|iidd",
          const_cast<char**>(kwlist),
          &candidates_by_consumed,
          &source_len,
          &beam_width,
          &max_words,
          &max_distance,
          &prune_ratio)) {
    return nullptr;
  }

  if (!PyList_Check(candidates_by_consumed)) {
    PyErr_SetString(PyExc_TypeError, "candidates_by_consumed must be a list");
    return nullptr;
  }

  std::vector<Node> nodes;
  nodes.reserve(1024);
  nodes.push_back(Node{-1, -1, 0, 0, 0.0, 0.0});

  std::vector<int> beam;
  beam.push_back(0);

  std::vector<int> complete_nodes;
  complete_nodes.reserve(256);

  double best_complete_score = std::numeric_limits<double>::infinity();
  double score_ceil = max_distance * prune_ratio;

  for (int round_i = 0; round_i < max_words; ++round_i) {
    if (beam.empty()) {
      break;
    }

    std::vector<int> next_beam;
    next_beam.reserve(static_cast<size_t>(beam_width) * 4);

    for (int node_idx : beam) {
      const Node& n = nodes[static_cast<size_t>(node_idx)];
      if (n.consumed >= source_len) {
        continue;
      }

      PyObject* cands = PyList_GetItem(candidates_by_consumed, n.consumed);
      if (!cands || !PyList_Check(cands)) {
        continue;
      }

      const Py_ssize_t cands_len = PyList_GET_SIZE(cands);
      for (Py_ssize_t ci = 0; ci < cands_len; ++ci) {
        PyObject* cand = PyList_GET_ITEM(cands, ci);
        if (!cand || !PyTuple_Check(cand) || PyTuple_GET_SIZE(cand) < 3) {
          continue;
        }

        const int n_tok = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(cand, 0)));
        const double seg_cost = PyFloat_AsDouble(PyTuple_GET_ITEM(cand, 1));
        const int entry_id = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(cand, 2)));

        int new_consumed = n.consumed + n_tok;
        if (new_consumed > source_len) {
          new_consumed = source_len;
        }

        const double new_raw = n.raw + seg_cost;
        const double new_score = (new_consumed > 0) ? (new_raw / static_cast<double>(new_consumed)) : new_raw;

        if (new_score > score_ceil) {
          continue;
        }

        const int new_depth = n.depth + 1;
        nodes.push_back(Node{node_idx, entry_id, new_consumed, new_depth, new_raw, new_score});
        const int new_idx = static_cast<int>(nodes.size() - 1);

        if (new_consumed >= source_len) {
          complete_nodes.push_back(new_idx);
          if (new_score < best_complete_score) {
            best_complete_score = new_score;
            score_ceil = best_complete_score * prune_ratio;
          }
        } else if (new_depth < max_words) {
          next_beam.push_back(new_idx);
        }
      }
    }

    if (static_cast<int>(next_beam.size()) > beam_width) {
      std::partial_sort(
          next_beam.begin(),
          next_beam.begin() + beam_width,
          next_beam.end(),
          [&nodes](int a, int b) { return nodes[static_cast<size_t>(a)].score < nodes[static_cast<size_t>(b)].score; });
      next_beam.resize(static_cast<size_t>(beam_width));
    }

    beam.swap(next_beam);
  }

  PyObject* node_parent = PyList_New(static_cast<Py_ssize_t>(nodes.size()));
  PyObject* node_entry = PyList_New(static_cast<Py_ssize_t>(nodes.size()));
  PyObject* node_raw = PyList_New(static_cast<Py_ssize_t>(nodes.size()));
  PyObject* node_score = PyList_New(static_cast<Py_ssize_t>(nodes.size()));
  PyObject* complete = PyList_New(static_cast<Py_ssize_t>(complete_nodes.size()));

  if (!node_parent || !node_entry || !node_raw || !node_score || !complete) {
    Py_XDECREF(node_parent);
    Py_XDECREF(node_entry);
    Py_XDECREF(node_raw);
    Py_XDECREF(node_score);
    Py_XDECREF(complete);
    return PyErr_NoMemory();
  }

  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(nodes.size()); ++i) {
    const Node& n = nodes[static_cast<size_t>(i)];
    PyList_SET_ITEM(node_parent, i, PyLong_FromLong(n.parent));
    PyList_SET_ITEM(node_entry, i, PyLong_FromLong(n.entry));
    PyList_SET_ITEM(node_raw, i, PyFloat_FromDouble(n.raw));
    PyList_SET_ITEM(node_score, i, PyFloat_FromDouble(n.score));
  }

  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(complete_nodes.size()); ++i) {
    PyList_SET_ITEM(complete, i, PyLong_FromLong(complete_nodes[static_cast<size_t>(i)]));
  }

  PyObject* out = PyTuple_New(5);
  if (!out) {
    Py_DECREF(node_parent);
    Py_DECREF(node_entry);
    Py_DECREF(node_raw);
    Py_DECREF(node_score);
    Py_DECREF(complete);
    return nullptr;
  }

  PyTuple_SET_ITEM(out, 0, node_parent);
  PyTuple_SET_ITEM(out, 1, node_entry);
  PyTuple_SET_ITEM(out, 2, node_raw);
  PyTuple_SET_ITEM(out, 3, node_score);
  PyTuple_SET_ITEM(out, 4, complete);
  return out;
}

PyObject* beam_rescore_paths_cpp(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
  PyObject* source_idx_arr = nullptr;
  int source_len = 0;
  PyObject* packed_paths = nullptr;
  PyObject* offsets_arr = nullptr;
  PyObject* all_tgt_idx_arr = nullptr;
  PyObject* dist_flat_arr = nullptr;
  int matrix_dim = 0;
  double max_distance = 0.50;

  static const char* kwlist[] = {
      "source_idx_arr",
      "source_len",
      "packed_paths",
      "offsets_arr",
      "all_tgt_idx_arr",
      "dist_flat_arr",
      "matrix_dim",
      "max_distance",
      nullptr};

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OiOOOOi|d",
          const_cast<char**>(kwlist),
          &source_idx_arr,
          &source_len,
          &packed_paths,
          &offsets_arr,
          &all_tgt_idx_arr,
          &dist_flat_arr,
          &matrix_dim,
          &max_distance)) {
    return nullptr;
  }

  if (source_len <= 0) {
    return PyList_New(0);
  }
  if (!PyList_Check(packed_paths)) {
    PyErr_SetString(PyExc_TypeError, "packed_paths must be a list");
    return nullptr;
  }

  Py_buffer src_buf;
  Py_buffer off_buf;
  Py_buffer tgt_buf;
  Py_buffer dist_buf;
  bool src_ok = false;
  bool off_ok = false;
  bool tgt_ok = false;
  bool dist_ok = false;

  if (PyObject_GetBuffer(source_idx_arr, &src_buf, PyBUF_SIMPLE) != 0) {
    return nullptr;
  }
  src_ok = true;
  if (PyObject_GetBuffer(offsets_arr, &off_buf, PyBUF_SIMPLE) != 0) {
    PyBuffer_Release(&src_buf);
    return nullptr;
  }
  off_ok = true;
  if (PyObject_GetBuffer(all_tgt_idx_arr, &tgt_buf, PyBUF_SIMPLE) != 0) {
    PyBuffer_Release(&off_buf);
    PyBuffer_Release(&src_buf);
    return nullptr;
  }
  tgt_ok = true;
  if (PyObject_GetBuffer(dist_flat_arr, &dist_buf, PyBUF_SIMPLE) != 0) {
    PyBuffer_Release(&tgt_buf);
    PyBuffer_Release(&off_buf);
    PyBuffer_Release(&src_buf);
    return nullptr;
  }
  dist_ok = true;

  const auto release_buffers = [&]() {
    if (dist_ok) {
      PyBuffer_Release(&dist_buf);
    }
    if (tgt_ok) {
      PyBuffer_Release(&tgt_buf);
    }
    if (off_ok) {
      PyBuffer_Release(&off_buf);
    }
    if (src_ok) {
      PyBuffer_Release(&src_buf);
    }
  };

  if (src_buf.len % static_cast<Py_ssize_t>(sizeof(Py_ssize_t)) != 0 ||
      tgt_buf.len % static_cast<Py_ssize_t>(sizeof(Py_ssize_t)) != 0 ||
      off_buf.len % static_cast<Py_ssize_t>(sizeof(int32_t)) != 0 ||
      dist_buf.len % static_cast<Py_ssize_t>(sizeof(double)) != 0) {
    release_buffers();
    PyErr_SetString(PyExc_TypeError, "unexpected array dtype/shape for beam_rescore_paths_cpp");
    return nullptr;
  }

  const auto* src = static_cast<const Py_ssize_t*>(src_buf.buf);
  const auto* offsets = static_cast<const int32_t*>(off_buf.buf);
  const auto* all_tgt = static_cast<const Py_ssize_t*>(tgt_buf.buf);
  const auto* dist_flat = static_cast<const double*>(dist_buf.buf);

  const Py_ssize_t src_count = src_buf.len / static_cast<Py_ssize_t>(sizeof(Py_ssize_t));
  const Py_ssize_t off_count = off_buf.len / static_cast<Py_ssize_t>(sizeof(int32_t));
  const Py_ssize_t tgt_count = tgt_buf.len / static_cast<Py_ssize_t>(sizeof(Py_ssize_t));
  const Py_ssize_t dist_count = dist_buf.len / static_cast<Py_ssize_t>(sizeof(double));

  if (src_count != source_len || off_count < 2 ||
      dist_count != static_cast<Py_ssize_t>(matrix_dim) * static_cast<Py_ssize_t>(matrix_dim)) {
    release_buffers();
    PyErr_SetString(PyExc_ValueError, "invalid shapes for beam_rescore_paths_cpp inputs");
    return nullptr;
  }

  const Py_ssize_t n_entries = off_count - 1;
  std::vector<RescoreResult> kept;
  kept.reserve(static_cast<size_t>(PyList_GET_SIZE(packed_paths)));

  const Py_ssize_t n_paths = PyList_GET_SIZE(packed_paths);
  for (Py_ssize_t i = 0; i < n_paths; ++i) {
    PyObject* path = PyList_GET_ITEM(packed_paths, i);
    if (!path || !PyTuple_Check(path) || PyTuple_GET_SIZE(path) < 3) {
      continue;
    }

    PyObject* entry_ids_obj = PyTuple_GET_ITEM(path, 1);
    const double raw_cost = PyFloat_AsDouble(PyTuple_GET_ITEM(path, 2));
    if (PyErr_Occurred()) {
      release_buffers();
      return nullptr;
    }

    PyObject* entry_seq = PySequence_Fast(entry_ids_obj, "entry_ids must be a sequence");
    if (!entry_seq) {
      release_buffers();
      return nullptr;
    }

    const Py_ssize_t n_ids = PySequence_Fast_GET_SIZE(entry_seq);
    PyObject** items = PySequence_Fast_ITEMS(entry_seq);

    std::vector<long> entry_ids;
    entry_ids.reserve(static_cast<size_t>(n_ids));
    Py_ssize_t total_len = 0;
    bool valid = true;

    for (Py_ssize_t j = 0; j < n_ids; ++j) {
      const long eid_l = PyLong_AsLong(items[j]);
      if (PyErr_Occurred()) {
        valid = false;
        break;
      }

      if (eid_l < 0 || static_cast<Py_ssize_t>(eid_l) >= n_entries) {
        valid = false;
        break;
      }

      const Py_ssize_t eid = static_cast<Py_ssize_t>(eid_l);
      const Py_ssize_t start = static_cast<Py_ssize_t>(offsets[eid]);
      const Py_ssize_t end = static_cast<Py_ssize_t>(offsets[eid + 1]);
      if (start < 0 || end < start || end > tgt_count) {
        valid = false;
        break;
      }

      total_len += (end - start);
      entry_ids.push_back(eid_l);
    }

    Py_DECREF(entry_seq);
    if (!valid || total_len <= 0) {
      if (PyErr_Occurred()) {
        release_buffers();
        return nullptr;
      }
      continue;
    }

    const Py_ssize_t denom = (source_len >= total_len) ? source_len : total_len;
    const Py_ssize_t len_diff = (source_len >= total_len) ? (source_len - total_len) : (total_len - source_len);
    if (denom > 0 && (static_cast<double>(len_diff) / static_cast<double>(denom)) > max_distance) {
      continue;
    }

    std::vector<Py_ssize_t> tgt;
    tgt.reserve(static_cast<size_t>(total_len));
    for (long eid_l : entry_ids) {
      const Py_ssize_t eid = static_cast<Py_ssize_t>(eid_l);
      const Py_ssize_t start = static_cast<Py_ssize_t>(offsets[eid]);
      const Py_ssize_t end = static_cast<Py_ssize_t>(offsets[eid + 1]);
      for (Py_ssize_t k = start; k < end; ++k) {
        tgt.push_back(all_tgt[k]);
      }
    }

    const double raw = dp_edit_distance(
        src,
        static_cast<Py_ssize_t>(source_len),
        tgt.data(),
        static_cast<Py_ssize_t>(tgt.size()),
        dist_flat,
        static_cast<Py_ssize_t>(matrix_dim));

    const double dist = raw / static_cast<double>(denom);
    if (dist <= max_distance) {
      kept.push_back(RescoreResult{dist, std::move(entry_ids), raw_cost});
    }
  }

  release_buffers();

  std::sort(
      kept.begin(),
      kept.end(),
      [](const RescoreResult& a, const RescoreResult& b) { return a.distance < b.distance; });

  PyObject* out = PyList_New(static_cast<Py_ssize_t>(kept.size()));
  if (!out) {
    return nullptr;
  }

  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(kept.size()); ++i) {
    const auto& r = kept[static_cast<size_t>(i)];

    PyObject* path_tup = PyTuple_New(static_cast<Py_ssize_t>(r.entry_ids.size()));
    if (!path_tup) {
      Py_DECREF(out);
      return nullptr;
    }
    for (Py_ssize_t j = 0; j < static_cast<Py_ssize_t>(r.entry_ids.size()); ++j) {
      PyObject* v = PyLong_FromLong(r.entry_ids[static_cast<size_t>(j)]);
      if (!v) {
        Py_DECREF(path_tup);
        Py_DECREF(out);
        return nullptr;
      }
      PyTuple_SET_ITEM(path_tup, j, v);
    }

    PyObject* tup = PyTuple_New(3);
    if (!tup) {
      Py_DECREF(path_tup);
      Py_DECREF(out);
      return nullptr;
    }

    PyObject* dist_obj = PyFloat_FromDouble(r.distance);
    PyObject* raw_obj = PyFloat_FromDouble(r.raw_cost);
    if (!dist_obj || !raw_obj) {
      Py_XDECREF(dist_obj);
      Py_XDECREF(raw_obj);
      Py_DECREF(path_tup);
      Py_DECREF(tup);
      Py_DECREF(out);
      return nullptr;
    }

    PyTuple_SET_ITEM(tup, 0, dist_obj);
    PyTuple_SET_ITEM(tup, 1, path_tup);
    PyTuple_SET_ITEM(tup, 2, raw_obj);
    PyList_SET_ITEM(out, i, tup);
  }

  return out;
}

PyMethodDef ModuleMethods[] = {
    {"beam_state_search_cpp",
     reinterpret_cast<PyCFunction>(beam_state_search_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "C++ beam state search."},
    {"beam_rescore_paths_cpp",
     reinterpret_cast<PyCFunction>(beam_rescore_paths_cpp),
     METH_VARARGS | METH_KEYWORDS,
     "C++ beam rescoring kernel."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_beam_cpp",
    "C++ acceleration for beam search state expansion.",
    -1,
    ModuleMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__beam_cpp(void) { return PyModule_Create(&ModuleDef); }
