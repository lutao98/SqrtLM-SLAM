#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "backend/mybackend/problem.h"
#include "utils/tic_toc.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {
namespace backend {

void Problem::LogoutVectorSize() {
    if(debug_output_)
    std::cout << "problem::LogoutVectorSize verticies_:" << verticies_.size() 
              << " edges:" << edges_.size();
}


Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    opetimize_level_ = 0;     //默认为0
    debug_output_ = false;
    draw_hessian_ = false;
    storageMode_ = StorageMode::GENERIC_MODE;
    forceStopFlag_ = nullptr;
    LogoutVectorSize();
}


Problem::~Problem() {
    if(debug_output_)    std::cout << "Problem is deleted!"<<std::endl;
    global_vertex_id = 0;
    global_edge_id = 0;
}


bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        std::cerr << "Vertex " << vertex->Id() << " has been added before!!";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }

    return true;
}


bool Problem::AddEdge(shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        std::cerr << "Edge " << edge->Id() << " has been added before!!";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }

    return true;
}


bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose");
}


bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ");
}


bool Problem::Solve(int iterations) {

    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;
    while ( !stop && (iter < iterations) && !terminate() ) {

        if(debug_output_)
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;     //尝试正确增量delta_x的次数

        while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步  // basalt里面也会这样
        {
            // setLambda,这个lambda是在大H里面加
            // 也可以再SC后的Hpp里面加(这样计算次数少,也不用减),但是测完发现不容易得出正确的增量
            AddLambdatoHessianLM();
            // 第四步，解线性方程，加了lambda
            SolveLinearSystem();
            RemoveLambdaHessianLM();    //因为不能保证当前lambda迭代一定成功,所以要减去

            // 优化退出条件1： delta_x_ 很小则退出
            // if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)
            // TODO:: 退出条件还是有问题, 好多次误差都没变化了，还在迭代计算，应该搞一个误差不变了就中止

            // 更新状态量
            UpdateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = IsGoodStepInLM();
            // 后续处理
            if (oneStepSuccess) {
                if(debug_output_)    std::cout << "get one step success\n";
                // 在新线性化点 构建 hessian
                MakeHessian();
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
        // TODO:: 应该改成前后两次的误差已经不再变化
        // if (sqrt(currentChi_) <= stopThresholdLM_)
        // if (sqrt(currentChi_) < 1e-15)
        if(last_chi_ - currentChi_ < 1e-5)
        {
            // std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
            stop = true;
        }
        last_chi_ = currentChi_;
    }

    if(debug_output_)
        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl
                  << "makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;

    t_hessian_cost_ = 0.;

    return true;
}


void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));  //从0开始,后面会改成接着pose的idx
    }
}


void Problem::SetOrdering() {

    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // 先分开统计pose和landmark的idx，然后把landmark的idx加上pose的数量(StorageMode::GENERIC_MODE)
    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数

        // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
        if (problemType_ == ProblemType::SLAM_PROBLEM)    
        {
            AddOrderingSLAM(vertex.second);
        }
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if(storageMode_ == StorageMode::GENERIC_MODE){
            // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
            ulong all_pose_dimension = ordering_poses_;
            for (auto landmarkVertex : idx_landmark_vertices_) {
                landmarkVertex.second->SetOrderingId(
                    landmarkVertex.second->OrderingId() + all_pose_dimension
                );
            }
        }else{
            // 如果是稠密的存储方式,Hll是单独管理的,则不需要
        }
    }

    if(storageMode_ == StorageMode::DENSE_MODE){
        Hll_.clear();
        Hll_.reserve( idx_landmark_vertices_.size() );
    }

//    CHECK_EQ(CheckOrdering(), true);
}


bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }

        for (auto v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}


void Problem::MakeHessian() {
    TicToc t_h;
    
    ulong size = ordering_generic_;
    b_=VecX::Zero(size);

    if(storageMode_ == StorageMode::GENERIC_MODE){    // 这种稀疏的存储方式在构造和拷贝大矩阵时会耗时

        // 直接构造大的 H 矩阵,如果是4000*4000,大概耗时30ms,而fill才5ms,copy也是30ms
        MatXX H(MatXX::Zero(size, size));

        // accelate, accelate, accelate
        #ifdef USE_OPENMP
        omp_set_dynamic(1);
        omp_set_num_threads(4);
        #pragma omp parallel for
        #endif

        for (auto iter:edges_) {                    // openmp并行不能写!=,还是不太优雅

            std::shared_ptr<Edge> e=iter.second;
            // auto iter = edges_.begin();          // vector/deque/string 是随机迭代器,支持+n  [n]  <=操作
            // for(int j=0; j<k; j++) iter++;       // map/list/set 是双向迭代器 只支持++ !=    而stack/queue不支持迭代器

            if(e->getLevel() != opetimize_level_) continue;    //只计算opetimize_level_的误差边,其他的算outlier

            // 取出每条边，也就是一次观测，计算雅可比和残差
            e->ComputeResidual();
            e->ComputeJacobians();

            // TODO:: robust cost
            auto jacobians = e->Jacobians();     // vector,每个雅可比维度是 residual x vertex[i]
            auto verticies = e->Verticies();     // 重投影误差edge的话,对应landmark和pose的vertex

            assert(jacobians.size() == verticies.size());

            for (size_t i = 0; i < verticies.size(); ++i) {

                auto v_i = verticies[i];

                if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto jacobian_i = jacobians[i];  // 对应顶点的雅可比
                ulong index_i = v_i->OrderingId();
                ulong dim_i = v_i->LocalDimension();   

                // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
                double drho;
                MatXX robustInfo(e->Information().rows(),e->Information().cols());
                e->RobustInfo(drho,robustInfo);

                MatXX JtW = jacobian_i.transpose() * robustInfo;     // Jt×W

                for (size_t j = i; j < verticies.size(); ++j) {

                    auto v_j = verticies[j];

                    if (v_j->IsFixed()) continue;

                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->OrderingId();
                    ulong dim_j = v_j->LocalDimension();

                    assert(v_j->OrderingId() != -1);

                    MatXX hessian = JtW * jacobian_j;   // Jt×J  并且算上robust

                    // 所有的信息矩阵叠加起来
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;    // noalias避免生成中间量,从而加速
                    if (j != i) {
                        // 对称的下三角
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b_.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* e->Information() * e->Residual();
            }
        }

        Hessian_ = H;                  // 如果H比较大,copy也耗时

        // 把H画出来
        if(draw_hessian_){
            cv::Mat HImage=cv::Mat::zeros(Hessian_.rows(),Hessian_.cols(),CV_8UC1);
            for(int i=0; i<Hessian_.rows(); i++){
                for(int j=0; j<Hessian_.cols(); j++){
                    if(Hessian_(i,j)!=0) HImage.at<uchar>(i,j)=255;
                }
            }
            cv::imshow("Hessian", HImage);
            cv::waitKey();
        }

    }else{  // 这种稠密存储方式在存Hll时避免了存储大量的零,从而计算和构造拷贝时都会快

        // | Hpp Hpl |
        // | Hlp Hll |
        Hpp_ = MatXX::Zero(ordering_poses_, ordering_poses_);
        Hpl_ = MatXX::Zero(ordering_poses_, ordering_landmarks_);

        int pose_size = idx_pose_vertices_.size(), landmark_size = idx_landmark_vertices_.size();

        if(Hll_.empty()){
            Hll_.resize( landmark_size, Mat33::Zero() );
        }else{                                              //不重新申请空间,而是直接赋值
            Mat33 zero(Mat33::Zero());
            for(Mat33 &hll:Hll_){
                hll=zero;
            }
        }

        for (auto iter:edges_) {

            std::shared_ptr<Edge> e=iter.second;

            if( e->getLevel() != opetimize_level_ ) continue;    //只计算opetimize_level_的误差边,其他的算outlier

            // 取出每条边，也就是一次观测，计算雅可比和残差
            e->ComputeResidual();
            e->ComputeJacobians();

            auto jacobians = e->Jacobians();     // vector,每个雅可比维度是 residual x vertex[i]
            auto verticies = e->Verticies();     // 重投影误差edge的话,对应landmark和pose的vertex

            assert(jacobians.size() == verticies.size());

            for (size_t i = 0; i < verticies.size(); ++i) {

                auto v_i = verticies[i];

                if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto jacobian_i = jacobians[i];  // 对应顶点的雅可比
                ulong index_i = v_i->OrderingId();
                ulong dim_i = v_i->LocalDimension();

                // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
                double drho;
                MatXX robustInfo(e->Information().rows(),e->Information().cols());
                e->RobustInfo(drho,robustInfo);

                MatXX JtW = jacobian_i.transpose() * robustInfo;  // Jt×W

                for (size_t j = i; j < verticies.size(); ++j) {

                    auto v_j = verticies[j];

                    if (v_j->IsFixed()) continue;

                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->OrderingId();
                    ulong dim_j = v_j->LocalDimension();

                    assert(v_j->OrderingId() != -1);

                    MatXX hessian = JtW * jacobian_j;   // Jt×J  并且算上robust
                    
                    // Hpp Hpl
                    // Hlp Hll
                    if( IsPoseVertex(v_i) && IsPoseVertex(v_j) )
                    { 
                        Hpp_.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;   // Hpp
                    }else if( IsPoseVertex(v_i)&&IsLandmarkVertex(v_j) )
                    {
                        Hpl_.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;   // Hpl
                    }else if(IsLandmarkVertex(v_i)&&IsLandmarkVertex(v_j))
                    {
                        // vertex的id是自动生成的,pose在前,landmark在后,所以需要减去pose的size
                        Hll_.at(v_j->Id()-pose_size) = hessian;                            // Hll
                    }else{
                        // Hlp不用存,Hpl转置
                    }
                }
                b_.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* e->Information() * e->Residual();
            }
        }

    }

    t_hessian_cost_ += t_h.toc();  //记录H构建的耗时

    delta_x_ = VecX::Zero(size);   // initial delta_x = 0_n;

}


/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
void Problem::SolveLinearSystem() {


    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        // PCG solver
        MatXX H = Hessian_;
        for (size_t i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
        delta_x_ = H.ldlt().solve(b_);

    } else {

        TicToc t_Hmminv;

        // step1: schur marginalization --> Hpp, bpp
        int reserve_size = ordering_poses_;
        int marg_size = ordering_landmarks_;

        if(marg_size == 0){    // 对于pose only optimization,不需要marg
            TicToc t_linearsolver;
            delta_x_ = Hessian_.ldlt().solve(b_);
            // if(debug_output_)   std::cout << "      Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;
            return ;
        }

        if(storageMode_ == StorageMode::GENERIC_MODE){

            // 下面耗时以marg_size为4000为例
            // copy耗时(marg_size比较大),大概35ms
            MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);

            // Hpm大概零点几,bpp更少
            MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
            MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
            VecX bpp = b_.segment(0, reserve_size);
            VecX bmm = b_.segment(reserve_size, marg_size);

            // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
            // 因为marg_size比较大,构造也很耗时,大概30ms
            MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));

            // 这个地方不怎么耗时,大概1ms
            for (auto landmarkVertex : idx_landmark_vertices_) {
                int idx = landmarkVertex.second->OrderingId() - reserve_size;
                int size = landmarkVertex.second->LocalDimension();
                Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
            }

            // 大概10ms
            MatXX tempH = Hpm * Hmm_inv;
            // H_pp_schur = Hpp - Hpl*Hll_inv*Hlp
            MatXX H_pp_schur = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
            // b_pp_schur = bp - Hpl*Hll_inv*bl
            VecX b_pp_schur = bpp - tempH * bmm;

            // step2: solve H_pp_schur * delta_x_pp = b_pp_schur
            VecX delta_x_pp(VecX::Zero(reserve_size));

            // // 这个lambda直接在S矩阵上加的,可以测测有什么不一样,H纬度比较大
            // // 测试:发现这种方法的lambda不容易得到正确的增量,从而会跳出迭代
            // for (ulong i = 0; i < ordering_poses_; ++i) {
            //     H_pp_schur_(i, i) += currentLambda_;              // LM Method
            // }

            // 求pose增量
            TicToc t_linearsolver;
            delta_x_pp =  H_pp_schur.ldlt().solve(b_pp_schur);       //  SVec.asDiagonal() * svd.matrixV() * Ub;    
            delta_x_.head(reserve_size) = delta_x_pp;
            // if(debug_output_)    std::cout << "      Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

            // 利用pose增量反求出landmark增量,大概5ms
            // step3: solve Hmm * delta_x_ll = bmm - Hmp * delta_x_pp;
            VecX delta_x_ll(marg_size);
            delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
            delta_x_.tail(marg_size) = delta_x_ll;

        }else{    // dense存储方式的SC求解

            MatXX Hpm = Hpl_;
            MatXX Hmp = Hpl_.transpose();
            VecX bpp = b_.segment(0, reserve_size);
            VecX bmm = b_.segment(reserve_size, marg_size);

            std::vector<Mat33> Hll_inv;
            Hll_inv.reserve(Hll_.size());
            for(const Mat33 &hll:Hll_){
                Hll_inv.emplace_back( hll.inverse() );
            }

            MatXX tempH(MatXX::Zero(reserve_size, marg_size));

            assert(marg_size == 3*Hll_inv.size());

            // 分块乘,避免大量的零块相乘
            for(int i=0; i<Hll_inv.size(); i++){
                tempH.block(0, 3*i, reserve_size, 3).noalias() = Hpm.block(0, 3*i, reserve_size, 3)*Hll_inv[i];
            }

            MatXX temp(MatXX::Zero(reserve_size, reserve_size));
            temp.noalias() = tempH * Hmp;                // 这个地方会有时候突然特别耗时,还不知道原因,原来0.几,突然要十几

            // H_pp_schur = Hpp - Hpl*Hll_inv*Hlp
            MatXX H_pp_schur = Hpp_ - temp;
            // b_pp_schur = bp - Hpl*Hll_inv*bl
            VecX b_pp_schur = bpp - tempH * bmm;

            // step2: solve H_pp_schur * delta_x_pp = b_pp_schur
            VecX delta_x_pp(VecX::Zero(reserve_size));

            // 求pose增量
            TicToc t_linearsolver;
            delta_x_pp =  H_pp_schur.ldlt().solve(b_pp_schur);   //  SVec.asDiagonal() * svd.matrixV() * Ub;    
            delta_x_.head(reserve_size) = delta_x_pp;
            // if(debug_output_)    std::cout << "      Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

            // 利用pose增量反求出landmark增量
            // step3: solve Hmm * delta_x_ll = bmm - Hmp * delta_x_pp;
            VecX delta_x_ll(marg_size);
            VecX indirect_vec = (bmm - Hmp * delta_x_pp);
            for(int i=0; i<Hll_inv.size(); i++){
                delta_x_ll.segment(3*i, 3).noalias() = Hll_inv[i]*indirect_vec.segment(3*i, 3);
            }
            delta_x_.tail(marg_size) = delta_x_ll;

        }

        if(debug_output_)
            std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
    }

}


void Problem::UpdateStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->BackUpParameters();    // 保存上次的估计值
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

}


void Problem::RollbackStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

}


/// LM
void Problem::ComputeLambdaInitLM() {
    // three P16/77
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }

    currentChi_ *= 0.5;

    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-10 倍

    double maxDiagonal = 0;

    if(storageMode_ == StorageMode::GENERIC_MODE){
        ulong size = Hessian_.cols();
        assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
        for (ulong i = 0; i < size; ++i) {
            maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
        }
    }else{     // dense方式存储是四个块分开管理
        ulong Hpp_size = Hpp_.cols();
        assert(Hpp_.rows() == Hpp_.cols() && "Hessian is not square");
        for (ulong i = 0; i < Hpp_size; ++i) {
            maxDiagonal = std::max(fabs(Hpp_(i, i)), maxDiagonal);
        }
        for(const Mat33 &hll:Hll_){
            for(int i=0; i<3; i++){
                maxDiagonal = std::max(fabs(hll(i, i)), maxDiagonal);
            }
        }
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau = 1e-5;  // 1e-5
    currentLambda_ = tau * maxDiagonal;    // 取对角线元素最大值的tau倍
    if(debug_output_)    std::cout << "maxDiagonal: " << maxDiagonal << " currentLamba_: " << currentLambda_ << std::endl;
}

void Problem::AddLambdatoHessianLM() {

    if(storageMode_ == StorageMode::GENERIC_MODE){
        ulong size = Hessian_.cols();
        assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
        for (ulong i = 0; i < size; ++i) {
            Hessian_(i, i) += currentLambda_;
        }
    }else{
        ulong Hpp_size = Hpp_.cols();
        assert(Hpp_.rows() == Hpp_.cols() && "Hessian is not square");
        for (ulong i = 0; i < Hpp_size; ++i) {
            Hpp_(i, i) += currentLambda_;
        }
        for(Mat33 &hll:Hll_){
            for(int i=0; i<3; i++){
                hll(i, i) += currentLambda_;
            }
        }
    }

}

void Problem::RemoveLambdaHessianLM() {
    if(storageMode_ == StorageMode::GENERIC_MODE){
        ulong size = Hessian_.cols();
        assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
        // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
        for (ulong i = 0; i < size; ++i) {
            Hessian_(i, i) -= currentLambda_;
        }
    }else{
        ulong Hpp_size = Hpp_.cols();
        assert(Hpp_.rows() == Hpp_.cols() && "Hessian is not square");
        for (ulong i = 0; i < Hpp_size; ++i) {
            Hpp_(i, i) -= currentLambda_;
        }
        for(Mat33 &hll:Hll_){
            for(int i=0; i<3; i++){
                hll(i, i) -= currentLambda_;
            }
        }
    }

}


bool Problem::IsGoodStepInLM() {
    double scale = 0;
    // three P17/77
    // 近似下降
    scale = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-6;    // make sure it's non-zero :)

    // recompute residuals after update state
    // 计算更新之后的残差  然后用之前的残差减掉这个，就是实际下降
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }

    tempChi *= 0.5;          // 1/2 * err^2

    // three P20/77
    // 实际下降/近似下降
    double rho = (currentChi_ - tempChi) / scale;
    if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;
        ni_ = 2;
        currentChi_ = tempChi;
        return true;
    } else {
        currentLambda_ *= ni_;
        ni_ *= 2;
        return false;
    }
}


bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        std::cerr << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        std::cerr <<  "The edge " << edge->Id() << " is not in the problem!!" << endl;
        return false;
    }

    edges_.erase(edge->Id());
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());     //unordered_multimap
    for (auto iter = range.first; iter != range.second; ++iter) {

        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}

}
}






