// SYSTEM INCLUDES
#include <pybind11/pybind11.h>
#include <torch/extension.h>


// C++ PROJECT INCLUDES
#include "trees/covertree.h"

namespace py = pybind11;


PYBIND11_MODULE(gmra_trees, m)
{

    py::class_<CoverNode, std::shared_ptr<CoverNode> >(m, "CoverNode")
        .def(py::init([](int64_t pt_idx) { return std::make_shared<CoverNode>(pt_idx); }))
        .def("add_child", &CoverNode::add_child)
        .def("get_children", &CoverNode::get_children);

    py::class_<CoverTree>(m, "CoverTree")
        .def(py::init<int64_t, float>(),
             "constructor",
             py::arg("max_scale") = 10,
             py::arg("base") = 2.0)
        .def("insert", &CoverTree::insert)
        .def("insert_pt", &CoverTree::insert_pt)
        .def_property_readonly("root", &CoverTree::get_root)
        .def_property_readonly("num_nodes", &CoverTree::get_num_nodes)
        .def_property_readonly("min_scale", &CoverTree::get_min_scale)
        .def_property_readonly("max_scale", &CoverTree::get_max_scale);
}
