#include "model.h"

namespace lucrezio_semantic_perception{

  Model::Model(const std::string &type_,
               const Eigen::Isometry3f &pose_,
               const Eigen::Vector3f &min_,
               const Eigen::Vector3f &max_):
    _type(type_),
    _pose(pose_),
    _min(min_),
    _max(max_){}
}
