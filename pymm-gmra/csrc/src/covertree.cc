// SYSTEM INCLUDES
#include <iostream>
#include <math.h>
#include <list>


// C++ PROJECT INCLUDES
#include "trees/covertree.h"


// using ALL = torch::indexing::Slice;


CoverNodePtr
lookup(std::list<CoverNodePtr> l, int64_t idx)
{
    for(auto it = l.begin(); it != l.end(); ++it)
    {
        if(idx == 0)
        {
            return *it;
        }
        idx--;
    }
    return nullptr;
}


CoverNode::~CoverNode()
{
    this->_children.clear();
}


void
CoverNode::add_child(CoverNodePtr child,
                     int64_t scale)
{
    auto it = this->_children.find(scale);
    if(it != this->_children.end())
    {
       it->second.insert(child);
    } else
    {
        std::unordered_set<CoverNodePtr> new_children = {child};
        this->_children.insert({scale, new_children});
    }
}

std::list<CoverNodePtr>
CoverNode::get_children(int64_t scale,
                        bool only_children)
{
    //std::cout << "CoverNode::get_children: enter" << std::endl;
    std::list<CoverNodePtr> children;
    if(!only_children)
    {
        children.push_back(this->shared_from_this());
    }
    auto it = this->_children.find(scale);
    if(it != this->_children.end())
    {
        //std::cout << "CoverNode::get_children: node: " << this->_pt_idx << "has "
        //      << it->second.size() << "children!" << std::endl;
        for(auto child: it->second)
        {
            children.push_back(child);
        }
    }

    //std::cout << "CoverNode::get_children: only_children: " << only_children
    //          << " len(children): " << children.size() << std::endl;
    //std::cout << "CoverNode::get_children: exit" << std::endl;
    return children;
}


torch::Tensor
CoverNode::get_subtree_idxs(int64_t max_scale)
{
    std::list<CoverNodePtr> nodes_at_current_scale = {this->shared_from_this()};
    std::unordered_set<int64_t> pt_idxs;

    for(int64_t scale = max_scale; scale >= 0; --scale)
    {
        std::list<CoverNodePtr> children;
        for(auto node: nodes_at_current_scale)
        {
            for(auto c: node->get_children(scale, false))
            {
                children.push_back(c);
                pt_idxs.insert(node->_pt_idx);
            }
        }

        nodes_at_current_scale = children;
    }

    return torch::tensor(std::vector<int64_t>(pt_idxs.begin(), pt_idxs.end()));
}


CoverTree::~CoverTree()
{
}


void
CoverTree::insert(torch::Tensor X)
{
    for(int64_t pt_idx = 0; pt_idx < X.size(0); ++pt_idx)
    {
        this->insert_pt(pt_idx, X);
    }
}

void
CoverTree::insert_pt(int64_t pt_idx,
                     torch::Tensor X)
{
    //std::cout << "CoverTree::insert_pt: inserting pt_idx: " << pt_idx << std::endl;
    if(!this->_root)
    {
        this->_root = std::make_shared<CoverNode>(pt_idx);
        this->_num_nodes++;
        //std::cout << "CoverTree::insert_pt: done" << std::endl;
        return;
    }

    const torch::Tensor& pt = X.index({pt_idx}); //, ALL()});
    // const torch::Tensor& pt = X[torch::tensor({pt_idx})];

    Q_TYPE Qi_p_ds = {{this->_root},
                      this->compute_distances(pt,
                        X.index({this->_root->_pt_idx}))}; //, ALL()}))};

    int64_t scale = this->_max_scale;
    CoverNodePtr parent = nullptr;
    int64_t pt_scale = -1;

    bool stop = false;
    while(!stop)
    {
        float radius = std::pow(this->_base, scale);

        Q_TYPE Q_p_ds = this->get_children_and_distances(pt, X, Qi_p_ds, scale);
        float min_dist = this->get_min_dist(Q_p_ds);

        if(min_dist == 0)
        {
            return;
        } else if(min_dist > radius)
        {
            stop = true;
        } else
        {
            if(this->get_min_dist(Qi_p_ds) <= radius)
            {
                torch::Tensor parent_indices = (std::get<1>(Qi_p_ds) <= radius).nonzero();
                int64_t choice_idx = torch::randint(parent_indices.size(0), {1})
                    .item<int64_t>();
                parent = lookup(std::get<0>(Qi_p_ds),
                                parent_indices[choice_idx].item<int64_t>());
                pt_scale = scale;
            }

            {
                torch::Tensor new_Qi_p_ds_mask = std::get<1>(Q_p_ds) <= radius;
                std::list<CoverNodePtr> new_Qi_p_ds;
                int64_t idx = 0;
                for(auto it = std::get<0>(Q_p_ds).begin(); it != std::get<0>(Q_p_ds).end();
                    ++it)
                {
                    if(new_Qi_p_ds_mask[idx].item<bool>())
                    {
                        new_Qi_p_ds.push_back(*it);
                    }
                    idx++;
                }
                Qi_p_ds = std::make_tuple(new_Qi_p_ds,
                                         std::get<1>(Q_p_ds).index({new_Qi_p_ds_mask}));
                // Qi_p_ds = std::make_tuple(new_Qi_p_ds,
                //                           std::get<1>(Q_p_ds)[new_Qi_p_ds_mask]);
            }
            scale -= 1;
        }
    }

    auto new_node = std::make_shared<CoverNode>(pt_idx);
    parent->add_child(new_node, pt_scale);
    this->_num_nodes++;
    this->_min_scale = std::min(this->_min_scale, pt_scale-1);

    //std::cout << "CoverTree::insert_pt: done" << std::endl;
}

Q_TYPE
CoverTree::get_children_and_distances(torch::Tensor pt,
                                      torch::Tensor X,
                                      const Q_TYPE& Qi_p_ds,
                                      int64_t scale)
{
    //std::cout << "CoverTree::get_children_and_distances: entering" << std::endl;

    std::list<CoverNodePtr> Q;
    std::list<int64_t> Q_idxs;
    for(auto node: std::get<0>(Qi_p_ds))
    {
        //std::cout << "CoverTree::get_children_and_distances: node->pt_idx: "
        //          << node->_pt_idx << std::endl;
        for(auto child : node->get_children(scale, true))
        {
            //std::cout << "CoverTree::get_children_and_distances: \tchild->pt_idx: "
            //          << child->_pt_idx << std::endl;
            Q.push_back(child);
            Q_idxs.push_back(child->_pt_idx);
        }
    }

    //std::cout << "CoverTree::get_children_and_distances: Q_idxs: {";
    //for(auto i : Q_idxs)
    //{
    //    std::cout << i << ",";
    //}
    //std::cout << "}" << std::endl;

    if(Q_idxs.size() > 0)
    {
        torch::Tensor X_pts = X.index({torch::tensor(std::vector<int64_t>(Q_idxs.begin(),
                                                                          Q_idxs.end()))});
                                             //,
                                             //ALL()});
        // torch::Tensor X_pts = X[torch::tensor(std::vector<int64_t>(Q_idxs.begin(),
        //                                                            Q_idxs.end()))];

        // now join them together
        // std::list<CoverNodePtr> joined;
        for(auto x: std::get<0>(Qi_p_ds))
        {
            Q.push_back(x);
        }

        torch::Tensor Q_dists = this->compute_distances(pt, X_pts).view({-1});
        torch::Tensor Qi_dists = std::get<1>(Qi_p_ds);

        //std::cout << "CoverTree::get_children_and_distances: Q_dists.sizes(): "
        //          << Q_dists.sizes() << std::endl;
        //std::cout << "CoverTree::get_children_and_distances: Qi_dists.sizes(): "
        //          << Qi_dists.sizes() << std::endl;
        torch::Tensor dists = torch::cat({Q_dists, Qi_dists}, 0).view({-1});
        //std::cout << "CoverTree::get_children_and_distances: exit" << std::endl;
        return std::make_tuple(Q, dists);
    }

    return Qi_p_ds;
}


torch::Tensor
CoverTree::compute_distances(torch::Tensor pt,
                             torch::Tensor X)
{
    //std::cout << "CoverTree::compute_distances: enter" << std::endl;
    //std::cout << "CoverTree::compute_distances: pt.sizes(): " << pt.sizes() << std::endl;
    //std::cout << "CoverTree::compute_distances: X.sizes(): " << X.sizes() << std::endl;
    torch::Tensor ds = torch::sqrt(torch::pow(torch::abs(X - pt.view({1,-1})), 2).sum(1));

    //std::cout << "CoverTree::compute_distances: dists: " << ds.view({1,-1}) << std::endl;
    //std::cout << "CoverTree::compute_distances: exit" << std::endl;
    return ds;
}

float
CoverTree::get_min_dist(const Q_TYPE& Qi_p_ds)
{
    const auto distances = std::get<1>(Qi_p_ds);
    torch::Tensor min_dist = torch::min(distances);
    return min_dist.cpu().item<float>();
}

