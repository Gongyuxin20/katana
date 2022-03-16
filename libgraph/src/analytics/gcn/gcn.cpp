/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include "katana/analytics/gcn/gcn.h"

#include "katana/ArrowRandomAccessBuilder.h"
#include "katana/Statistics.h"
#include "katana/TypedPropertyGraph.h"
#include "katana/analytics/Utils.h"
#include "katana/PODVector.h"
#include "math.h"
#include "stdlib.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <map>
#include <set>
#include <string>


using namespace katana::analytics;

const int GCNPlan::kChunkSize = 64;

/*******************************************************************************
 * Functions for running the algorithm
 ******************************************************************************/
//! Node deadness can be derived from current degree and k value, so no field
//! necessary.
struct GCNNodeCurrentDegree : public katana::AtomicPODProperty<uint32_t> {};
struct GCNNodeLabel : public katana::AtomicPODProperty<uint8_t> {};
struct GCNNodeFeature : public katana::AtomicPODProperty<double> {};

struct GCNNodeAlive : public katana::PODProperty<uint32_t> {};

using NodeData = std::tuple<GCNNodeCurrentDegree>;
using EdgeData = std::tuple<>;

std::vector <std::vector < std::vector <double>>> W; // 权重矩阵
std::vector <std::vector < std::vector <double>>> dW; // W对应需要更新的梯度

std::vector <std::vector <double>> y; // label p维
std::vector <std::string> labels; // labels string类型
std::vector <std::vector <double>> pred_y; // pred_result p维
std::vector <double> pred_labels; // labels string类型
std::vector <std::vector <double>> dH1; // H1 处loss m维
std::vector <std::vector <double>> dH2; // H2 处loss p维

std::vector <std::vector < std::vector <double>>> feature; // Hi
std::vector <std::vector < std::vector <double>>> agg_weight; // Zi
std::vector <std::vector < std::vector <double>>> agg; // agg
std::vector <std::vector < std::vector <double>>> w0agg; // g'/W0 after agg
std::vector <std::vector < std::vector <double>>> w0agg_agg; // g'/W0 after agg_agg

void ReadContentFile(   std::string& content_file, 
                        std::vector <std::vector < std::vector <double>>>& feature, 
                        std::vector <std::string>& labels, const uint layers_plus )
{
  // 读取文件
  std::vector <std::vector <std::string>> info;
  std::ifstream infile;
	infile.open(content_file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
	std::vector<std::string> data;
	std::string s;
	while (std::getline(infile, s)) {
		std::istringstream is(s); //将读出的一行转成数据流进行操作
		std::string d;
		while (!is.eof()) {
			is >> d;
			data.push_back(d);
		}
		info.push_back(data);
		data.clear();
		s.clear();
	}
	infile.close();             //关闭文件输入流 

  // 提取元素
  for (uint i = 0; i < info.size(); i++ ){
      labels[i] = info[i][info[0].size()-1];
      feature[i].resize(layers_plus);
      feature[i][0].resize(info[0].size()-2);
      for(uint j = 1; j < info[0].size()-1; j++ ){
          std::istringstream iss(info[i][j]);  
          double num;  
          iss >> num;  
          feature[i][0][j-1]= num;
          
      }
      
  }
  
  // for (uint i = 0; i < feature.size(); i++ ){
  //     for(uint j = 0; j < feature[0][0].size(); j++ ){
  //         std::cout << feature[i][0][j] << " ";
  //     }
  //     std::cout << std::endl;
  // }
  std::cout <<"Reading Content Files is Sccuess"<< std::endl;


}
void OneHot( std::vector <std::string>& labels, std::vector <std::vector <double>>& y)
{
    //one_hot 编码
    std::map <std::string,uint> mp;
    std::set <std::string> classes; // classes string类型 看有多少类
    for (auto label : labels){
        classes.insert(label);
    }
    uint num = 0;
    for(std::set<std::string>::iterator it=classes.begin() ;it!=classes.end();it++)
    {
        mp.insert(std::make_pair(*it,num++));
    }
    std::map <std::string,uint>::iterator iter;
    iter = mp.begin();
    while(iter != mp.end()) {
        std::cout << iter->first << " : " << iter->second << std::endl;
        iter++;
    }

    for( uint src = 0; src < labels.size(); src++) {
       y[src].resize(mp.size(),0);
       std::string y_true = labels[src];
       y[src][mp[y_true]] = 1;
    }

    // for (auto a : y) {
	// 	for (auto  yy: a) {
	// 	std::cout << yy << " ";
    //     }
		
	// 	std::cout << std::endl;
	// }
}

void Normalize( std::vector <double>& v)
{
    //归一化
    double sum = 0.0000;
    for (uint i = 0; i < v.size(); i++) {
    sum += v[i];
    }
    for (uint i = 0; i < v.size(); i++) {
    v[i] = v[i]/sum;
    }

}

void AllocateMemory( uint64_t numNodes)
{
  y.resize(numNodes);
  pred_y.resize(numNodes);
  dH1.resize(numNodes);
  dH2.resize(numNodes);
  feature.resize(numNodes);
  agg_weight.resize(numNodes);
  agg.resize(numNodes);
  w0agg.resize(numNodes);
  w0agg_agg.resize(numNodes);
  labels.resize(numNodes);
  pred_labels.resize(numNodes);
    
}

void FreeMemory()
{
  y.resize(0);
  pred_y.resize(0);
  dH1.resize(0);
  dH2.resize(0);
  feature.resize(0);
  agg_weight.resize(0);
  agg.resize(0);
  w0agg.resize(0);
  w0agg_agg.resize(0);
  labels.resize(0);
  pred_labels.resize(0);
    
}

std::vector<double> dot( std::vector<double> v1, std::vector<std::vector<double> > v2)
{
  std::vector<double> v3(v2[0].size());
  double s = 0.0000;
  for (uint h = 0; h < v2[0].size(); h++){
    for (uint j = 0; j < v2.size(); j++){
      s += v1[j] * v2[j][h];
    }
    v3[h] = s;
    s =0.0000;
  }
  return v3;
}

std::vector<double> dot( std::vector<std::vector<double> > v2, std::vector<double> v1)
{
  std::vector<double> v3(v2.size());
  double s = 0.0000;
  for (uint h = 0; h < v2.size(); h++){
    for (uint j = 0; j < v2[0].size(); j++){
      s += v1[j] * v2[h][j];
    }
    v3[h] = s;
    s =0.0000;
  }
  return v3;
}

std::vector<double> ReLU( std::vector<double> v)
{
  // 实现relu激活函数
  std::vector<double> v3(v.size());
  for (uint i = 0; i < v.size(); i++){
    if (v[i] < 0){
      v3[i] = 0;
    }
    else{
      v3[i] = v[i];
    }
  }
  return v3;
}

std::vector<double> softmax(std::vector<double> v)
{
  // 实现softmax分类预测
  std::vector<double> v3(v.size());
  double sum = 0.0000;
  for (uint i = 0; i < v.size(); i++) {
    v3[i] =exp(v[i]);
    sum += exp(v[i]);
  }
  for (uint i = 0; i < v.size(); i++) {
    v3[i] = v3[i]/sum;
  }
  return v3;
}

double accuracy(std::vector<std::vector<double> > y_pred, std::vector<std::vector<double> > y_true)
{
  // 计算准确性
  double sum = y_pred.size();
  katana::GAccumulator<double> GAccumulator_result;
  katana::do_all(
      katana::iterate(static_cast<size_t>(0), y_pred.size()),
      [&](size_t i) {
        double max = 0;
        double max_idx = 0;
        double max_idy = 0;
        for (uint j = 0; j < y_pred[0].size(); j++){
          if (y_pred[i][j] > max){
            max = y_pred[i][j];
            max_idx = j;
          }
          if (y_true[i][j] == 1){
            max_idy = j;
          }
        }
        pred_labels[i]= max_idx;
        if (max_idx == max_idy){
          GAccumulator_result += 1.0;
        }
        
      },
      katana::loopname("Accuracy"), katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
    double result = GAccumulator_result.reduce();
    result = result/sum;
    return result;
}

/**
 * Initialize degree fields in graph with current degree. Since symmetric,
 * out edge count is equivalent to in-edge count.
 *
 * @param graph Graph to initialize degrees in
 */
template <typename GraphTy>
void
DegreeCounting(GraphTy* graph) {
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& node) {
        auto& node_current_degree =
            graph->template GetData<GCNNodeCurrentDegree>(node);
        node_current_degree.store(Degree(*graph, node));
      },
      katana::loopname("DegreeCounting"), 
      katana::no_stats());
}

/**
 * Initialize node state
 *
 * @param graph Graph to initialize in
 */
template <typename GraphTy>
void
InitializeNodeState(GraphTy* graph, const std::vector<double>& layers) {
  uint layer = layers.size() -1;
  W.resize(layer);
  dW.resize(layer);
  srand((unsigned)time(NULL));
  for (uint k = 0; k < layer; k++){
    W[k].resize(layers[k+1],std::vector<double>(layers[k],0));
    dW[k].resize(layers[k+1],std::vector<double>(layers[k],0));
    
    katana::do_all(
      katana::iterate(static_cast<size_t>(0), W[k].size()),
      [&](size_t i) {
        for (uint j =0; j < W[k][0].size(); j++) {
          W[k][i][j] = 2.0 * rand() / double(RAND_MAX) - 1.0;
          
        }
      },
      katana::loopname("RandomInitVector"), katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
  }
  uint layer_plus = layers.size();

  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& src) {
        w0agg[src].resize(layers[1], std::vector<double>(layers[0],0));
        // feature[src].resize(layer_plus, std::vector<double>(layers[0],0));
        
        agg_weight[src].resize(layer_plus);
        agg[src].resize(layer_plus);
        pred_y[src].resize(layers[2]);
        // y[src].resize(layers[2]);
        dH1[src].resize(layers[1]);
        dH2[src].resize(layers[2]);

        // feature[src][0][0] = src*0.01;
        // feature[src][0][1] = 1-src*0.01;

        // uint y1 = 1;
        // y[src][0] = y1;
        // y[src][1] = 1-y1;

      },
      katana::loopname("init_node_state"),katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
  // y[0][0] = 0;
  // y[0][1] = 1;
  // y[4][0] = 0;
  // y[4][1] = 1;
  // y[5][0] = 0;
  // y[5][1] = 1;
  // y[8][0] = 0;
  // y[8][1] = 1;
  // feature[0][0][0]=1;
  // feature[0][0][1]=0;
  // feature[4][0][0]=0.96;
  // feature[4][0][1]=0.04;
  // feature[5][0][0]=0.95;
  // feature[5][0][1]=0.05;
  // feature[8][0][0]=0.92;
  // feature[8][0][1]=0.08;
  // std::cout<<"y"<<std::endl;
  // for (GNode src=0; src< graph->NumNodes(); src++){
  //   std::cout<<src<<":::"<<y[src][0]<<" "<<y[src][1]<<std::endl;
  // }

}

/**
 * Activation Relu Function
 *
 * @param graph Graph to use
 */
template <typename GraphTy>
void
ActivationRelu(GraphTy* graph, const uint32_t& layerI) {
  // relu 激活
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& src) {
        auto& srcAGG = agg_weight[src][layerI];
        feature[src][layerI] = ReLU(srcAGG); // relu(agg * W)

      },
      katana::loopname("node_relu"),katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());

}

/**
 * Activation Softmax Function
 *
 * @param graph Graph to use
 */
template <typename GraphTy>
void
ActivationSoftmax(GraphTy* graph, const uint32_t& layerI) {
  // softmax 激活
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& src) {
        auto& srcAGG = agg_weight[src][layerI];
        feature[src][layerI] = softmax(srcAGG); // softmax(agg * W)
        pred_y[src] = feature[src][layerI]; // 最后一层目标函数

      },
      katana::loopname("node_softmax"),katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());

}

/**
 * GCN Forward Function
 *
 * @param graph Graph to use
 */
template <typename GraphTy>
void
Forward(GraphTy* graph, const uint32_t& layerI) {
  // softmax 激活
  using GNode = typename GraphTy::Node;
  // AGG
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& src) {
        double srcDegree = graph->template GetData<GCNNodeCurrentDegree>(src);
        auto& srcAGG = feature[src][layerI];
        auto& dstAGG = agg[src][layerI];
        dstAGG = srcAGG;
        for (auto e : Edges(*graph, src)) {
            auto dst = EdgeDst(*graph, e);
            auto& dstFeature = feature[dst][layerI];
            auto& dstDegree = 
                graph->template GetData<GCNNodeCurrentDegree>(dst);
            for (uint i = 0; i < srcAGG.size(); i++){
              dstAGG[i] = dstAGG[i] + (1/(sqrt(srcDegree * dstDegree))) * dstFeature[i];
            }
        }

      },
      katana::loopname("node_agg"),katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
  // std::cout<<"AGG"<<layerI<<std::endl;
  // for (GNode src = 0; src < graph->NumNodes(); src++){
  //   std::cout<<src<<":::"<<feature[src][layerI][0]<<" "<<feature[src][layerI][1]<<std::endl;
  // }
  // dot
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& src) {
        auto& srcAGG = agg[src][layerI];
        agg_weight[src][layerI+1] = dot (W[layerI], srcAGG); // W * agg     

      },
      katana::loopname("node_dot_w"),katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
  
}

/**
 * Calculate Loss Function
 *
 * @param graph Graph to use
 */
template <typename GraphTy>
void
ComputeLoss(GraphTy* graph, std::vector<double>& LOSS, const uint32_t& curRound) {
  // loss 计算
  katana::GAccumulator<double> GAccumulator_sum;
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& src) {
        for (uint i = 0; i < pred_y[src].size(); i++){
          GAccumulator_sum += -(y[src][i] * log(pred_y[src][i]));
        }

      },
      katana::loopname("node_loss"),katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
  double loss = GAccumulator_sum.reduce();
  std::cout<<"loss"<<":::"<<loss / graph->NumNodes()<<std::endl;
  LOSS[curRound] = loss / graph->NumNodes();
}

/**
 * Update Weight Function
 *
 */
void
UpdateWeight(const double& lr, const uint32_t& layerI) {
  // update weight
  
  katana::do_all(
      katana::iterate(static_cast<size_t>(0), W[layerI].size()),
      [&](size_t i) {
        for (uint j =0; j < W[layerI][0].size(); j++) {
          W[layerI][i][j] = W[layerI][i][j] - lr * dW[layerI][i][j];
        }
      },
      katana::loopname("UpdateVector"), katana::steal(), 
      katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
  
}

/**
 * Setup initial worklist of dead nodes.
 *
 * @param graph Graph to operate on
 * @param initial_worklist Empty worklist to be filled with dead nodes.
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number.
 */
template <typename GraphTy>
void
SetupInitialWorklist(
    const GraphTy& graph,
    katana::InsertBag<typename GraphTy::Node>& initial_worklist,
    uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(graph),
      [&](const GNode& node) {
        const auto& node_current_degree =
            graph.template GetData<GCNNodeCurrentDegree>(node);
        if (node_current_degree < k_core_number) {
          //! Dead node, add to initial_worklist for processing later.
          initial_worklist.emplace(node);
        }
      },
      katana::loopname("InitialWorklistSetup"), katana::no_stats());
}


/**
 * Compute GCN Algorithm
 *
 * @param graph Graph to operate on
 */
template <typename GraphTy>
void
ComputeGCN(GraphTy* graph, std::string& content_file) {
  
  std::vector<double> layers;
  layers.resize(3,0);
  
  ReadContentFile(content_file, feature, labels, layers.size());
  
  OneHot(labels, y);
  for (uint src =0 ; src < graph->NumNodes(); src++){
    Normalize(feature[src][0]);
  }
  // 记得改
  layers[0]=feature[0][0].size();
  layers[1]=16;
  layers[2]=y[0].size();
  std::cout << layers[0]<<std::endl;
  std::cout << layers[1]<<std::endl;
  std::cout << layers[2]<<std::endl;
  InitializeNodeState(graph, layers);
  // for (uint i = 0; i < feature.size(); i++ ){
  //       for(uint j = 0; j < feature[0][0].size(); j++ ){
  //           std::cout << feature[i][0][j] << " ";
  //       }
  //       std::cout << std::endl;
  //   }
  
  
  
  int curRound = 0;
  // double lr = 0.01/graph->NumNodes();
  
  // double lr = 0.000005;
  double lr = 0.01;
  uint k = 100;
  std::vector<double> LOSS;
  std::vector<double> ACCURACY;
  LOSS.resize(k,0);
  ACCURACY.resize(k,0);
  katana::StatTimer convergeTimer("converge_time-GCN");
  convergeTimer.start();
  
  using GNode = typename GraphTy::Node;
  while ( k-- ){
    // if (curRound % 50 == 0){
    //   lr =lr -0.000;
    // }
    std::cout<<"this is "<<curRound<<std::endl;
    Forward(graph, 0);
    ActivationRelu(graph, 1);
    Forward(graph, 1);
    ActivationSoftmax(graph, 2);
    ComputeLoss(graph, LOSS, curRound);
    double acc = accuracy(pred_y, y);
    ACCURACY[curRound] = acc;
    std::cout<<"accuracy"<<":::"<<acc<<std::endl;

    // 获取dW[1]
    // std::cout<<"size0 dh2 "<<dH2.size()<<std::endl;
    // std::cout<<"size1 dh2 "<<dH2[0].size()<<std::endl;
    // std::cout<<"size0 dw1 "<<dW[1].size()<<std::endl;
    // std::cout<<"size1 dw1 "<<dW[1][0].size()<<std::endl;
    // std::cout<<"size0 pred_y "<<pred_y.size()<<std::endl;
    // std::cout<<"size1 pred_y "<<pred_y[0].size()<<std::endl;
    // std::cout<<"size0 y "<<y.size()<<std::endl;
    // std::cout<<"size1 y "<<y[0].size()<<std::endl;
    // std::cout<<"size0 agg "<<agg.size()<<std::endl;
    // std::cout<<"size1 agg "<<agg[0].size()<<std::endl;
    // std::cout<<"size2 agg "<<agg[0][0].size()<<std::endl;
   
    katana::GAccumulator<double> GAccumulator_w1;
    for (uint i = 0; i < dW[1].size(); i++){
      for(uint j = 0; j < dW[1][0].size(); j++){
        GAccumulator_w1.reset();
        katana::do_all(
            katana::iterate(*graph),
            [&](const GNode& src) {
              dH2[src][i] = pred_y[src][i]-y[src][i];
              
              GAccumulator_w1 += dH2[src][i] * agg[src][1][j];
            },
            katana::loopname("node_dw1"),katana::steal(), 
            katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
        // dW[1][i][j] = GAccumulator_w1.reduce() / graph -> NumNodes();
        dW[1][i][j] = GAccumulator_w1.reduce();
        // std::cout<<i<<"::"<<j<<"dw1:::"<<GAccumulator_w1.reduce()<<" ";
        // std::cout<<i<<"::"<<j<<"dH2:::"<<GAccumulator_w1.reduce()<<":::"<<std::endl;
      }
      // std::cout<<std::endl;
    }
    

    // 两次汇聚第0层结果
    // AGG_AGG
    katana::do_all(
        katana::iterate(*graph),
        [&](const GNode& src) {
          for (uint i = 0; i < w0agg[src].size(); i++){
            for (uint j = 0; j < w0agg[src][0].size(); j++){
              if (agg_weight[src][1][i] >= 0){
                w0agg[src][i][j] = agg[src][0][j];
              }
            }
          }
        },
        katana::loopname("node_agg_agg0"),katana::steal(), 
        katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());

    
    katana::do_all(
        katana::iterate(*graph),
        [&](const GNode& src) {
          auto& srcAGG = w0agg[src];
          double srcDegree = 
                graph->template GetData<GCNNodeCurrentDegree>(src);
          auto& dstAGG = w0agg_agg[src]; // agg_agg
          dstAGG = srcAGG;
          for (auto e : Edges(*graph, src)) {
            auto dst = EdgeDst(*graph, e);
            auto& dstFeature = w0agg[dst];
            auto& dstDegree = 
                graph->template GetData<GCNNodeCurrentDegree>(dst);
            for (uint i = 0; i < srcAGG.size(); i++){
              for (uint j = 0; j < srcAGG[0].size(); j++){
                dstAGG[i][j] = dstAGG[i][j] + (1/(sqrt(srcDegree * dstDegree))) * dstFeature[i][j];
                
              }
            }
        }
        // loss 在层中传递
        dH1[src] = dot( dH2[src], W[1]);
        },
        katana::loopname("node_agg_agg0"),katana::steal(), 
        katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
    
    
    // 获取dW[0]
    katana::GAccumulator<double> GAccumulator_w0;
    for (uint i = 0; i < dW[0].size(); i++){
      for (uint j = 0; j < dW[0][0].size(); j++){
        GAccumulator_w0.reset();
        katana::do_all(
          katana::iterate(*graph),
          [&](const GNode& src) {
            GAccumulator_w0 += dH1[src][i] * w0agg_agg[src][i][j];
          },
          katana::loopname("node_dw0"),katana::steal(), 
          katana::chunk_size<GCNPlan::kChunkSize>(), katana::no_stats());
        // dW[0][i][j] = GAccumulator_w0.reduce()/ graph -> NumNodes();
        dW[0][i][j] = GAccumulator_w0.reduce();

      }
    }
    // 更新W
    UpdateWeight(lr, 1);
    UpdateWeight(lr, 0);
    curRound++;
    
  }

  convergeTimer.stop();
  double total_time = double(convergeTimer.get_usec())/1000000.0;
  
  for (uint i = 0; i < LOSS.size(); i++){
    std::cout<<"loss "<<i<<":::"<< LOSS[i]<<" "<<"accuracy"<<":::"<<ACCURACY[i]<<std::endl;
    
  }
  // std::cout<<"pred_labels"<<std::endl;
  // for (GNode src = 0; src < graph->NumNodes(); src++){
  //   std::cout<<src<<":::";
  //   std::cout<<pred_labels[src]<<" ";
  // }
  std::cout<<std::endl;
  std::cout<<"total convergeTimer :: "<<total_time<<"s"<<std::endl;
  FreeMemory();
}

/**
 * Starting with initial dead nodes, decrement degree and add to worklist
 * as they drop below 'k' threshold until worklist is empty (i.e. no more dead
 * nodes).
 *
 * @param graph Graph to operate on
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number.
 */
template <typename GraphTy>
void
AsyncCascadeGCN(GraphTy* graph, uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  katana::InsertBag<GNode> initial_worklist;
  //! Setup worklist.
  SetupInitialWorklist(*graph, initial_worklist, k_core_number);

  katana::for_each(
      katana::iterate(initial_worklist),
      [&](const GNode& dead_node, auto& ctx) {
        //! Decrement degree of all neighbors.
        for (auto e : Edges(*graph, dead_node)) {
          auto dest = EdgeDst(*graph, e);
          auto& dest_current_degree =
              graph->template GetData<GCNNodeCurrentDegree>(dest);
          uint32_t old_degree = katana::atomicSub(dest_current_degree, 1u);

          if (old_degree == k_core_number) {
            //! This thread was responsible for putting degree of destination
            //! below threshold: add to worklist.
            ctx.push(dest);
          }
        }
      },
      katana::disable_conflict_detection(),
      katana::chunk_size<GCNPlan::kChunkSize>(),
      katana::loopname("GCN Asynchronous"));
}

/**
 * After computation is finished, the nodes left in the core
 * are marked as alive.
 *
 * @param graph Graph to operate on
 * @param k_core_number Each node in the core is expected to have degree <= k_core_number.
 */
template <typename GraphTy>
katana::Result<void>
GCNMarkAliveNodes(GraphTy* graph, uint32_t k_core_number) {
  using GNode = typename GraphTy::Node;
  katana::do_all(
      katana::iterate(*graph),
      [&](const GNode& node) {
        auto& node_current_degree =
            graph->template GetData<GCNNodeCurrentDegree>(node);
        auto& node_flag = graph->template GetData<GCNNodeAlive>(node);
        node_flag = 1;
        if (node_current_degree < k_core_number) {
          node_flag = 0;
        }
      },
      katana::loopname("GCN Mark Nodes in Core"));
  return katana::ResultSuccess();
}

template <typename GraphTy>
static katana::Result<void>
GCNImpl(GraphTy* graph, std::string& content_file, GCNPlan algo, uint32_t k_core_number) {
  size_t approxNodeData = 4 * (graph->NumNodes() + graph->NumEdges());
  katana::EnsurePreallocated(8, approxNodeData);
  katana::ReportPageAllocGuard page_alloc;
  
  AllocateMemory(graph->NumNodes());
  //! Intialization of degrees.
  DegreeCounting(graph);

  //! Begins main computation.
  katana::StatTimer exec_time("GCN");

  exec_time.start();

  switch (algo.algorithm()) {
  case GCNPlan::kSynchronous:
    ComputeGCN(graph, content_file);
    break;
  case GCNPlan::kAsynchronous:
    AsyncCascadeGCN(graph, k_core_number);
    break;
  default:
    return katana::ErrorCode::AssertionFailed;
  }
  exec_time.stop();


  return katana::ResultSuccess();
}

katana::Result<void>
katana::analytics::GCN(
    katana::PropertyGraph* pg, std::string& content_file, uint32_t k_core_number,
    const std::string& output_property_name, katana::TxnContext* txn_ctx,
    const bool& is_symmetric, GCNPlan plan) {
  katana::analytics::TemporaryPropertyGuard temporary_property{
      pg->NodeMutablePropertyView()};

  KATANA_CHECKED(
      pg->ConstructNodeProperties<std::tuple<GCNNodeCurrentDegree>>(
          txn_ctx, {temporary_property.name()}));

  if (is_symmetric) {
    using Graph = katana::TypedPropertyGraphView<
        katana::PropertyGraphViews::Default, NodeData, EdgeData>;
    Graph graph =
        KATANA_CHECKED(Graph::Make(pg, {temporary_property.name()}, {}));

    KATANA_CHECKED(GCNImpl(&graph, content_file, plan, k_core_number));
  } else {
    using Graph = katana::TypedPropertyGraphView<
        katana::PropertyGraphViews::Undirected, NodeData, EdgeData>;

    Graph graph =
        KATANA_CHECKED(Graph::Make(pg, {temporary_property.name()}, {}));

    KATANA_CHECKED(GCNImpl(&graph, content_file, plan, k_core_number));
  }
  // Post processing. Mark alive nodes.
  KATANA_CHECKED(pg->ConstructNodeProperties<std::tuple<GCNNodeAlive>>(
      txn_ctx, {output_property_name}));

  using GraphTy = katana::TypedPropertyGraph<
      std::tuple<GCNNodeAlive, GCNNodeCurrentDegree>, std::tuple<>>;
  auto graph_final = KATANA_CHECKED(
      GraphTy::Make(pg, {output_property_name, temporary_property.name()}, {}));

  return GCNMarkAliveNodes(&graph_final, k_core_number);
}

// Doxygen doesn't correctly handle implementation annotations that do not
// appear in the declaration.
/// \cond DO_NOT_DOCUMENT
// TODO (gill) Add a validity routine.
katana::Result<void>
katana::analytics::GCNAssertValid(
    [[maybe_unused]] katana::PropertyGraph* pg,
    [[maybe_unused]] uint32_t k_core_number,
    [[maybe_unused]] const std::string& property_name) {
  return katana::ResultSuccess();
}

katana::Result<GCNStatistics>
katana::analytics::GCNStatistics::Compute(
    katana::PropertyGraph* pg, [[maybe_unused]] uint32_t k_core_number,
    const std::string& property_name) {
  using Graph =
      katana::TypedPropertyGraph<std::tuple<GCNNodeAlive>, std::tuple<>>;
  using GNode = Graph::Node;
  auto pg_result = Graph::Make(pg, {property_name}, {});
  if (!pg_result) {
    return pg_result.error();
  }

  auto graph = pg_result.value();

  katana::GAccumulator<uint32_t> alive_nodes;
  alive_nodes.reset();

  katana::do_all(
      katana::iterate(graph),
      [&](const GNode& node) {
        auto& node_alive = graph.GetData<GCNNodeAlive>(node);
        if (node_alive) {
          alive_nodes += 1;
        }
      },
      katana::loopname("GCN sanity check"), katana::no_stats());

  return GCNStatistics{alive_nodes.reduce()};
}
/// \endcond DO_NOT_DOCUMENT

void
katana::analytics::GCNStatistics::Print(std::ostream& os) const {
  os << "Number of nodes in the core = " << number_of_nodes_in_kcore
     << std::endl;
}
