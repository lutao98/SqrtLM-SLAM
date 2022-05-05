#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>

#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

typedef unsigned long ulong;

namespace myslam {
namespace backend {

typedef unsigned long ulong;
// typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;   // vertex需要按照id排序遍历,所以不能用这个
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
public:

    /**
     * 问题的类型
     * SLAM问题还是通用的问题
     *
     * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏/稠密方式存储
     * SLAM问题只接受一些特定的Vertex和Edge
     * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
     */
    enum class ProblemType {
        SLAM_PROBLEM,
        GENERIC_PROBLEM
    };

    /**
     * @brief SLAM问题H矩阵的存储方式
     * 
     */
    enum class StorageMode {
        GENERIC_MODE,
        DENSE_MODE             // 稠密存储方式,会快
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Problem(ProblemType problemType);

    ~Problem();

    bool AddVertex(std::shared_ptr<Vertex> vertex);

    /**
     * remove a vertex
     * @param vertex_to_remove
     */
    bool RemoveVertex(std::shared_ptr<Vertex> vertex);

    bool AddEdge(std::shared_ptr<Edge> edge);

    bool RemoveEdge(std::shared_ptr<Edge> edge);

    /**
     * 取得在优化中被判断为outlier部分的边，方便前端去除outlier
     * @param outlier_edges
     */
    void GetOutlierEdges(std::vector<std::shared_ptr<Edge>> &outlier_edges);

    /**
     * 求解此问题
     * @param iterations
     * @return
     */
    bool Solve(int iterations = 10);

    void setOptimizeLevel(int opetimize_level) { opetimize_level_= opetimize_level; }

    int getEdgeSize() {return edges_.size();}

    void setDebugOutput(bool debug) { debug_output_ = debug; }

    void setForceStopFlag(bool *flag) { forceStopFlag_ = flag; }

    void setStorageMode(StorageMode storageMode) { storageMode_ = storageMode; }

    void setDrawHessian(bool draw_hessian) { draw_hessian_ = draw_hessian; }

private:

    /// 设置各顶点的ordering_index
    void SetOrdering();

    /// set ordering for new vertex in slam problem
    void AddOrderingSLAM(std::shared_ptr<Vertex> v);

    /// 构造大H矩阵
    void MakeHessian();

    /// 解线性方程
    void SolveLinearSystem();

    /// 更新状态变量
    void UpdateStates();

    /// 有时候 update 后残差会变大，需要退回去，重来
    void RollbackStates();

    /// 判断一个顶点是否为Pose顶点
    bool IsPoseVertex(std::shared_ptr<Vertex> v);

    /// 判断一个顶点是否为landmark顶点
    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

    /// 检查ordering是否正确
    bool CheckOrdering();

    void LogoutVectorSize();

    /// 获取某个顶点连接到的边
    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

    /// Levenberg
    /// 计算LM算法的初始Lambda
    void ComputeLambdaInitLM();

    /// Hessian 对角线加上或者减去 Lambda
    void AddLambdatoHessianLM();

    void RemoveLambdaHessianLM();

    /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    bool IsGoodStepInLM();

    /// PCG 迭代线性求解器
    VecX PCGSolver(const MatXX &A, const VecX &b, int maxIter);

    bool terminate() {return forceStopFlag_ ? (*forceStopFlag_) : false; }

    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    double ni_;                 //控制 Lambda 缩放大小

    ProblemType problemType_;
    StorageMode storageMode_;

    /// 整个信息矩阵,一般存储方式
    MatXX Hessian_;
    VecX b_;

    /// 信息矩阵稠密存储方式,Hll以vector形式存储,避免存0
    MatXX Hpp_;                 // 稀疏,但是维度不大
    MatXX Hpl_;                 // 这一块根据具体观测决定是稀疏的还是稠密的,不需要存Hlp
    std::vector<Mat33> Hll_;    // 稀疏,维度一般特别大,XYZ参数化

    VecX delta_x_;

    /// all vertices
    HashVertex verticies_;
    /// all edges
    HashEdge edges_;
    /// 由vertex id查询edge,只在删除vertex的时候用到
    HashVertexIdToEdge vertexToEdge_;

    /// Ordering related
    ulong ordering_poses_ = 0;
    ulong ordering_landmarks_ = 0;
    ulong ordering_generic_ = 0;
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // 以ordering排序的pose顶点
    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // 以ordering排序的landmark顶点

    double t_hessian_cost_ = 0.0;
    double t_PCGsovle_cost_ = 0.0;

    int opetimize_level_;     // 优化属性,在第opetimize_level_的边才构造约束,在多轮优化时可以用到

    bool debug_output_;
    bool draw_hessian_;

    bool *forceStopFlag_;     // 外界设置的停止优化标志,用指针传值

};

}
}

#endif
