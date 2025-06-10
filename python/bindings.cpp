#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "qg/qg.hpp"
#include "qg/qg_builder.hpp"

namespace py = pybind11;
using py_float_array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_uint_array = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

namespace {
void get_arr_shape(const py::buffer_info& buffer, size_t& rows, size_t& cols) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        std::cerr << "Input data has an incorrect shape. Data must be a 1D or 2D array.\n";
        return;
    }
    if (buffer.ndim == 2) {
        rows = buffer.shape[0];
        cols = buffer.shape[1];
    } else {
        rows = 1;
        cols = buffer.shape[0];
    }
}
}  // namespace

struct Index {
    std::unique_ptr<symqg::QuantizedGraph> index = nullptr;

    explicit Index(
        const std::string& index_type,
        const std::string& metric,
        size_t num_points,
        size_t dim,
        size_t degree
    ) {
        if (metric != "L2") {
            std::cerr << "Only L2 distance supported currently\n";
            throw std::invalid_argument("Only L2 distance supported currently");
        }

        if (degree < 32 || degree % 32 != 0) {
            std::cerr << "The degree bound must be a multiple of 32\n";
            throw std::invalid_argument("The degree bound must be a multiple of 32");
        }

        if (index_type == "QG") {
            index = std::make_unique<symqg::QuantizedGraph>(num_points, degree, dim);
        } else {
            std::cerr << "Index type [" << index_type << "] not supported\n";
            throw std::invalid_argument("Index type [" + index_type + "] not supported");   
        }
    }

    void load(const std::string& filename) const { index->load_index(filename.c_str()); }

    void save(const std::string& filename) const { index->save_index(filename.c_str()); }

    void set_ef(size_t ef_search) const { index->set_ef(ef_search); }

    void set_init_ef(size_t ef_init) const { index->set_init_ef(ef_init); }

    void set_num_inserted_threshold(size_t threshold) const { index->set_num_inserted_threshold(threshold); }

    void build_index(
        const py::object& data,
        size_t ef_indexing,
        size_t num_iter = 3,
        size_t num_threads = UINT_MAX
    ) const {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
        auto buffer = items.request();
        size_t num = 0;
        size_t dim = 0;
        get_arr_shape(buffer, num, dim);
        if (num != index->num_vertices() || dim != index->dimension()) {
            std::cerr
                << "The shape of data is different with initialization! Expected shape: ("
                << index->num_vertices() << ", " << index->dimension() << "), but got: ("
                << num << ", " << dim << ")\n";
            throw std::invalid_argument("The shape of data is different with initialization! Expected shape: (" + std::to_string(index->num_vertices()) + ", " + std::to_string(index->dimension()) + "), but got: (" + std::to_string(num) + ", " + std::to_string(dim) + ")");
        }
        symqg::QGBuilder builder(*index, ef_indexing, items.data(), num_threads);
        builder.build(num_iter);
        std::cout << "\tQuantizedGraph created\n";
    }

    auto search(py_float_array& query, uint32_t knn) const {
        py_uint_array result(knn);
        auto* result_ptr = static_cast<uint32_t*>(result.request().ptr);
        index->search(query.data(0), knn, result_ptr);

        return result;
    }
};

PYBIND11_MODULE(symphonyqg, m) {
    m.doc() = R"pbdoc(Towards Symphonious Integration of Graph and Quantization)pbdoc";

    py::class_<Index>(m, "Index")
        .def(
            py::init<const std::string&, const std::string&, size_t, size_t, size_t>(),
            py::arg("index_type"),
            py::arg("metric"),
            py::arg("num_elements"),
            py::arg("dimension"),
            py::arg("degree_bound") = 32
        )
        .def("load", &Index::load, py::arg("filename"))
        .def("save", &Index::save, py::arg("filename"))
        .def("set_ef", &Index::set_ef, py::arg("EF"))
        .def("set_init_ef", &Index::set_init_ef, py::arg("EF"))
        .def("set_num_inserted_threshold", &Index::set_num_inserted_threshold, py::arg("threshold"))
        .def(
            "build_index",
            &Index::build_index,
            py::arg("data"),
            py::arg("EF"),
            py::arg("num_iter") = 3,
            py::arg("num_threads") = UINT_MAX
        )
        .def("search", &Index::search, py::arg("query"), py::arg("k"));
}
